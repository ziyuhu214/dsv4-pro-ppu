"""B-class A/B v2: FlagGems vs the implementation the stack actually runs.

v1 compared against torch or against vllm._C, which was wrong on both counts:
vllm._C does not exist in this stack (its .so is parked in /tmp/so_park so that
plugin-fl registers its own schemas), and a torch per-group mm loop is not the
vendor grouped kernel. The real comparison side is the `deep_gemm` vendor library
reached through plugin-fl's wrappers -- that is what produces
`cutlass::deep_gemm::GemmKernel<...>` and `Sm80PagedMqaLogits` in the trace.

Calling conventions taken from plugin-fl's own call sites
(vllm_fl/ops/ppu_deep_gemm_moe.py:408-453): bf16 grouped passes plain tensors,
int8 grouped passes (tensor, scale) tuples, weights are `nt` layout (E, N, K),
m_indices=expert_ids, m_rows=experts_for_rows.

Shapes from DeepSeek-V4-Pro config.json at tp=16/ep=32: hidden 7168,
moe_intermediate 3072, 12 experts/rank, W8A8 int8.
"""
import argparse
import json
import os
import traceback

os.environ.setdefault("FLAGGEMS_ENABLE_OPLIST_PATH", "/tmp/bclass2_oplist.txt")

import torch
from torch.profiler import ProfilerActivity, profile

OUT = "/tmp/bclass_ab2.jsonl"
N_ITERS = 20
d = "cuda"
H, MOE_INTER, E = 7168, 3072, 12


def rec(**kw):
    with open(OUT, "a") as f:
        f.write(json.dumps(kw, ensure_ascii=False, default=str) + "\n")
    print("  " + json.dumps({k: v for k, v in kw.items() if k != "trace"},
                            ensure_ascii=False, default=str)[:230])


def measure(fn, iters=N_ITERS):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    tot, kers = 0.0, {}
    for e in p.key_averages():
        sd = getattr(e, "self_device_time_total", 0) or 0
        if sd > 0 and not e.key.startswith("aten::") and not e.key.startswith("cuda"):
            kers[e.key[:95]] = round(sd / iters, 2)
            tot += sd
    s, ev = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    ev.record()
    torch.cuda.synchronize()
    return (round(tot / iters, 2), round(s.elapsed_time(ev) * 1000.0 / iters, 2),
            dict(sorted(kers.items(), key=lambda kv: -kv[1])[:3]))


def attempt(op, side, call):
    try:
        out = call()
        torch.cuda.synchronize()
    except Exception as e:
        rec(op=op, side=side, stage="call", error=f"{type(e).__name__}: {e}"[:260],
            trace=traceback.format_exc()[-500:])
        return None
    try:
        k, w, kers = measure(call)
        rec(op=op, side=side, kernel_us=k, wall_us=w, kernels=kers)
    except Exception as e:
        rec(op=op, side=side, stage="measure", error=f"{type(e).__name__}: {e}"[:260])
    return out


def cmp(op, a, b):
    if a is None or b is None:
        return
    a = a[0] if isinstance(a, (tuple, list)) else a
    b = b[0] if isinstance(b, (tuple, list)) else b
    if a.shape != b.shape:
        rec(op=op, side="diff", note=f"shape {tuple(a.shape)} vs {tuple(b.shape)}")
        return
    rec(op=op, side="diff",
        max_abs_diff=(a.float() - b.float()).abs().max().item(),
        ref_absmax=b.float().abs().max().item())


# =============================================================== MoE grouped bf16
def moe_grouped(tokens):
    """FlagGems group_mm vs deep_gemm m_grouped_bf16_gemm_nt_nopad."""
    g = torch.Generator().manual_seed(11)
    K, N = H, MOE_INTER
    per = tokens // E
    A = torch.randn(tokens, K, generator=g).to(torch.bfloat16).to(d)
    # deep_gemm wants nt: (E, N, K). FlagGems group_mm wants (E, K, N).
    W_nt = torch.randn(E, N, K, generator=g).to(torch.bfloat16).to(d)
    W_kn = W_nt.transpose(1, 2).contiguous()

    m_indices = torch.repeat_interleave(
        torch.arange(E, dtype=torch.int32, device=d),
        torch.full((E,), per, dtype=torch.int32, device=d))
    if m_indices.numel() < tokens:
        m_indices = torch.cat([m_indices, torch.full(
            (tokens - m_indices.numel(),), E - 1, dtype=torch.int32, device=d)])
    m_rows = torch.full((E,), per, dtype=torch.int32, device=d)
    m_rows[-1] = tokens - per * (E - 1)
    # FlagGems offs = cumulative end offsets
    offs = torch.cumsum(m_rows, 0).to(torch.int32)

    import flag_gems
    from vllm_fl.ops.ppu_deep_gemm import m_grouped_bf16_gemm_nt_nopad

    out_dg = torch.empty(tokens, N, dtype=torch.bfloat16, device=d)

    op = f"moe_grouped_gemm_bf16[t={tokens},E={E},K={K},N={N}]"
    og = attempt(op, "flaggems_group_mm", lambda: flag_gems.group_mm(A, W_kn, offs))
    od = attempt(op, "deep_gemm_nt_nopad",
                 lambda: (m_grouped_bf16_gemm_nt_nopad(
                     A, W_nt, out_dg, m_indices, m_rows), out_dg)[1])
    cmp(op, og, od)


# ============================================================ paged MQA logits
def mqa_logits(bs):
    """FlagGems bf16_paged_mqa_logits vs deep_gemm's, same signature."""
    import deep_gemm
    import flag_gems

    heads, dim = 64, 128          # index_n_heads, index_head_dim
    block_kv, seq = 64, 4096
    nblocks = seq // block_kv

    g = torch.Generator().manual_seed(23)
    q = torch.randn(bs, 1, heads, dim, generator=g).to(torch.bfloat16).to(d)
    kv_cache = torch.randn(nblocks * bs, block_kv, 1, dim,
                           generator=g).to(torch.bfloat16).to(d)
    weights = torch.randn(bs * 1, heads, generator=g).to(torch.float32).to(d)
    # deep_gemm requires 2D context_lens (B, next_n) -- csrc/apis/attention.hpp,
    # per the comment at vllm/v1/attention/backends/mla/indexer.py:608-612. A 1D
    # tensor trips a bare assert inside the extension.
    ctx2d = torch.full((bs, 1), seq, dtype=torch.int32, device=d)
    bt = torch.arange(nblocks * bs, dtype=torch.int32, device=d).reshape(bs, nblocks)

    try:
        # deep_gemm asserts (schedule_meta.shape[0] - 1) % get_num_sms() == 0
        # (jit_kernels/attention.py:451). On PPU get_num_sms() is 20 while torch
        # reports multi_processor_count == 64, so the torch value fails.
        num_sms = deep_gemm.get_num_sms()
        meta = deep_gemm.get_paged_mqa_logits_metadata(ctx2d, block_kv, num_sms)
    except Exception as e:
        rec(op=f"paged_mqa_logits[bs={bs}]", side="metadata",
            error=f"{type(e).__name__}: {e}"[:260])
        return

    op = f"paged_mqa_logits_bf16[bs={bs},h={heads},d={dim},seq={seq}]"
    og = attempt(op, "flaggems", lambda: flag_gems.bf16_paged_mqa_logits(
        q, kv_cache, weights, ctx2d, bt, meta, seq))
    od = attempt(op, "deep_gemm", lambda: deep_gemm.bf16_paged_mqa_logits(
        q, kv_cache, weights, ctx2d, bt, meta, seq))
    cmp(op, og, od)


# ================================================================== int8 GEMM
def int8_gemm(M):
    """FlagGems cutlass_scaled_mm (int8) vs deep_gemm gemm_int8_int8_bf16_nt."""
    import deep_gemm
    import flag_gems

    K, N = H, 2048
    g = torch.Generator().manual_seed(29)
    a = torch.randint(-8, 8, (M, K), generator=g, dtype=torch.int8).to(d)
    w_nt = torch.randint(-8, 8, (N, K), generator=g, dtype=torch.int8).to(d)
    asc = torch.rand(M, 1, generator=g, dtype=torch.float32).to(d)
    bsc = torch.rand(N, 1, generator=g, dtype=torch.float32).to(d)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=d)

    op = f"int8_gemm[M={M},K={K},N={N}]"

    def gems():
        c = torch.empty(M, N, dtype=torch.bfloat16, device=d)
        # op wants b as (K, N) with b.stride(0) == 1, i.e. column-major.
        # w_nt is (N, K) contiguous, so w_nt.t() is exactly that -- no copy.
        b_cm = w_nt.t()
        flag_gems.cutlass_scaled_mm(c, a, b_cm, asc.squeeze(1), bsc.squeeze(1), None)
        return c

    og = attempt(op, "flaggems_cutlass_scaled_mm", gems)
    od = attempt(op, "deep_gemm_int8_nt",
                 lambda: (deep_gemm.gemm_int8_int8_bf16_nt(
                     (a, asc), (w_nt, bsc), out), out)[1])
    cmp(op, og, od)


# ========================================================== sparse MLA prefill
def mla_sparse(tokens, hq=8):
    """FlagGems flash_mla_sparse_fwd vs the vendor flash_mla package.

    Shapes per FlagGems' own docstring (fused/flashmla_sparse.py:1058-1060):
    q [s_q, h_q, d_qk], kv [s_kv, h_kv, d_qk], indices [s_q, h_kv, topk] -- the
    indices tensor is 3D with an h_kv axis, which is what v1 got wrong.
    """
    import flash_mla as vendor

    import flag_gems

    # FlagGems asserts HQ in (64, 128) at fused/flashmla_sparse.py:1107, so the
    # real tp=16 shape (128/16 = 8 heads per rank) cannot run at all. hq is
    # parameterised so the 8-head (on-config, FlagGems unusable) and 64-head
    # (off-config, both run) cases can both be recorded.
    dqk, dv = 576, 512
    topk, kv_len = 1024, 4096          # index_topk from config
    g = torch.Generator().manual_seed(13)
    q = torch.randn(tokens, hq, dqk, generator=g).to(torch.bfloat16).to(d).contiguous()
    kv = torch.randn(kv_len, 1, dqk, generator=g).to(torch.bfloat16).to(d).contiguous()
    idx = torch.stack([
        torch.randperm(kv_len, generator=g)[:topk].sort().values for _ in range(tokens)
    ]).to(torch.int32).unsqueeze(1).to(d).contiguous()   # [s_q, h_kv=1, topk]
    scale = dqk ** -0.5

    op = f"flash_mla_sparse_fwd[sq={tokens},hq={hq},topk={topk},skv={kv_len}]"
    og = attempt(op, "flaggems",
                 lambda: flag_gems.flash_mla_sparse_fwd(q, kv, idx, scale, dv))
    ov = attempt(op, "vendor_flash_mla",
                 lambda: vendor.flash_mla_sparse_fwd(q, kv, idx, scale, dv))
    cmp(op, og, ov)


# ============================================================== MLA decode
def mla_decode(bs):
    """FlagGems flash_mla vs vendor flash_mla_with_kvcache."""
    import flash_mla as vendor

    import flag_gems

    hq, hkv, dqk, dv = 8, 1, 576, 512
    block, seq = 64, 4096
    nblocks = seq // block
    g = torch.Generator().manual_seed(17)
    q = torch.randn(bs, 1, hq, dqk, generator=g).to(torch.bfloat16).to(d).contiguous()
    blocked_k = torch.randn(nblocks * bs, block, hkv, dqk,
                            generator=g).to(torch.bfloat16).to(d).contiguous()
    bt = torch.arange(nblocks * bs, dtype=torch.int32, device=d).reshape(bs, nblocks)
    cs = torch.full((bs,), seq, dtype=torch.int32, device=d)

    op = f"flash_mla_decode[bs={bs},hq={hq},seq={seq}]"
    og = attempt(op, "flaggems",
                 lambda: flag_gems.flash_mla(q, bt, blocked_k, seq, block, bs, 1,
                                             cs, hq, hkv, dqk, dv, True))

    def vend():
        meta, splits = vendor.get_mla_metadata(cs, 1 * hq // hkv, hkv)
        return vendor.flash_mla_with_kvcache(
            q, blocked_k, bt, cs, dv,
            tile_scheduler_metadata=meta, num_splits=splits, causal=True)

    ov = attempt(op, "vendor_flash_mla", vend)
    cmp(op, og, ov)


CASES = {
    "moe": lambda: [moe_grouped(t) for t in (192, 1024, 4096)],
    "mqa": lambda: [mqa_logits(b) for b in (64, 256)],
    "int8": lambda: [int8_gemm(m) for m in (512, 4096)],
    "sparse": lambda: [mla_sparse(t) for t in (64, 1024)],
    "sparse64": lambda: [mla_sparse(t, hq=64) for t in (64, 1024)],
    "decode": lambda: [mla_decode(b) for b in (64, 256)],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=sorted(CASES))
    a = ap.parse_args()
    print(f"===== {a.case} =====")
    CASES[a.case]()
