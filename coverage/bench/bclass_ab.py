"""B-class A/B: FlagGems implementations vs what the stack actually runs.

B-class = operators whose FlagGems source exists but is unreachable in this stack
because apply_gems_patches_to_vllm() has no call site. The question here is
whether wiring them up would even be worth it, so each one is called directly,
bypassing the missing patch.

Discipline carried over from the mHC harness:
  * one record per measurement, appended to JSONL immediately
  * every op is attempted and its failure recorded verbatim -- "would not run on
    this hardware" is a result, not a gap in the report
  * device kernel time via profiler (not CUDA events around one call, which for
    small kernels measures Triton dispatch instead)
  * numerical comparison whenever both sides produce a tensor

Shapes from DeepSeek-V4-Pro config.json at tp=16:
  hidden 7168, 128 heads -> 8 heads/rank, head_dim 512, qk_rope 64,
  index_n_heads 64, index_head_dim 128, index_topk 1024,
  384 experts -> 12/rank (ep=32), moe_intermediate 3072, W8A8 int8.
"""
import argparse
import json
import os
import traceback

os.environ.setdefault("FLAGGEMS_ENABLE_OPLIST_PATH", "/tmp/bclass_oplist.txt")

import torch
from torch.profiler import ProfilerActivity, profile

OUT = "/tmp/bclass_ab.jsonl"
N_ITERS = 20
d = "cuda"

H = 7168
HC = 4
HEADS_PER_RANK = 8
HEAD_DIM = 512
QK_ROPE = 64
IDX_HEADS = 64
IDX_DIM = 128
IDX_TOPK = 1024
EXPERTS_PER_RANK = 12
MOE_INTER = 3072


def rec(**kw):
    with open(OUT, "a") as f:
        f.write(json.dumps(kw, ensure_ascii=False, default=str) + "\n")
    print("  " + json.dumps({k: v for k, v in kw.items() if k != "trace"},
                            ensure_ascii=False, default=str)[:220])


def kernel_us(fn, iters=N_ITERS):
    """Device kernel time per call, plus the kernel symbols that ran."""
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
            kers[e.key[:100]] = round(sd / iters, 2)
            tot += sd
    # wall
    s, ev = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    ev.record()
    torch.cuda.synchronize()
    return (round(tot / iters, 2), round(s.elapsed_time(ev) * 1000.0 / iters, 2),
            dict(sorted(kers.items(), key=lambda kv: -kv[1])[:3]))


def diff(a, b):
    if a is None or b is None:
        return None
    if isinstance(a, (tuple, list)):
        return max((diff(x, y) or 0.0) for x, y in zip(a, b))
    if a.shape != b.shape:
        return float("inf")
    return (a.float() - b.float()).abs().max().item()


def attempt(op, side, build, call):
    """Build inputs and measure one side. Records failure verbatim."""
    try:
        args = build()
    except Exception as e:
        rec(op=op, side=side, stage="build", error=f"{type(e).__name__}: {e}"[:300])
        return None
    try:
        out = call(args)
        torch.cuda.synchronize()
    except Exception as e:
        rec(op=op, side=side, stage="call", error=f"{type(e).__name__}: {e}"[:300],
            trace=traceback.format_exc()[-600:])
        return None
    try:
        k, w, kers = kernel_us(lambda: call(args))
        rec(op=op, side=side, kernel_us=k, wall_us=w, kernels=kers)
        return out
    except Exception as e:
        rec(op=op, side=side, stage="measure", error=f"{type(e).__name__}: {e}"[:300])
        return out


# ---------------------------------------------------------------- hc_head_fused
def hc_head(tokens):
    def build():
        g = torch.Generator().manual_seed(7)
        hs = torch.randn(tokens, HC, H, generator=g).to(torch.bfloat16).to(d)
        fn = torch.randn(HC, HC * H, generator=g).to(torch.float32).to(d)
        sc = torch.rand(1, generator=g).to(torch.float32).to(d)
        ba = torch.randn(HC, generator=g).to(torch.float32).to(d)
        return hs, fn, sc, ba

    import flag_gems

    def gems(a):
        hs, fn, sc, ba = a
        out = torch.empty(tokens, H, dtype=torch.bfloat16, device=d)
        return flag_gems.hc_head_fused_kernel(hs, fn, sc, ba, out, H, 1e-6, 1e-6, HC)

    from vllm.model_executor.kernels.mhc.tilelang import hc_head_fused_kernel_tilelang

    def tl(a):
        hs, fn, sc, ba = a
        return hc_head_fused_kernel_tilelang(hs, fn, sc, ba, 1e-6, 1e-6)

    og = attempt(f"hc_head_fused[t={tokens}]", "flaggems", build, gems)
    ot = attempt(f"hc_head_fused[t={tokens}]", "vllm_tilelang", build, tl)
    if og is not None and ot is not None:
        rec(op=f"hc_head_fused[t={tokens}]", side="diff", max_abs_diff=diff(og, ot))


# ------------------------------------------------------------------- group_mm
def group_mm(tokens):
    """MoE grouped GEMM. FlagGems group_mm vs deep_gemm grouped kernel."""
    K, N, G = H, MOE_INTER, EXPERTS_PER_RANK

    def build():
        g = torch.Generator().manual_seed(11)
        A = torch.randn(tokens, K, generator=g).to(torch.bfloat16).to(d)
        B = torch.randn(G, K, N, generator=g).to(torch.bfloat16).to(d)
        per = tokens // G
        offs = torch.tensor([per * (i + 1) for i in range(G)],
                            dtype=torch.int32, device=d)
        offs[-1] = tokens
        return A, B, offs

    import flag_gems

    def gems(a):
        A, B, offs = a
        return flag_gems.group_mm(A, B, offs)

    def ref(a):
        # torch reference: per-group mm. Not the vendor deep_gemm kernel (that
        # one needs fp8/int8 layouts and a real MoE dispatch), so this is a
        # correctness+scale reference, labelled as such.
        A, B, offs = a
        outs = []
        start = 0
        for i in range(B.shape[0]):
            end = int(offs[i].item())
            if end > start:
                outs.append(torch.mm(A[start:end], B[i]))
            start = end
        return torch.cat(outs, 0) if outs else A.new_empty(0, B.shape[2])

    og = attempt(f"group_mm[t={tokens},G={G}]", "flaggems", build, gems)
    ot = attempt(f"group_mm[t={tokens},G={G}]", "torch_per_group_mm", build, ref)
    if og is not None and ot is not None:
        rec(op=f"group_mm[t={tokens},G={G}]", side="diff", max_abs_diff=diff(og, ot))


# --------------------------------------------------------- flash_mla_sparse_fwd
def mla_sparse(tokens):
    hq, dqk, dv = HEADS_PER_RANK, 576, 512
    topk = IDX_TOPK
    kv_len = 4096

    def build():
        g = torch.Generator().manual_seed(13)
        q = torch.randn(tokens, hq, dqk, generator=g).to(torch.bfloat16).to(d)
        kv = torch.randn(kv_len, 1, dqk, generator=g).to(torch.bfloat16).to(d)
        idx = torch.stack([
            torch.randperm(kv_len, generator=g)[:topk].sort().values
            for _ in range(tokens)
        ]).to(torch.int32).to(d)
        return q, kv, idx

    import flag_gems

    def gems(a):
        q, kv, idx = a
        return flag_gems.flash_mla_sparse_fwd(q, kv, idx, dqk ** -0.5, dv)

    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd as vfwd

    def vllm_(a):
        q, kv, idx = a
        return vfwd(q, kv, idx, dqk ** -0.5, dv)

    og = attempt(f"flash_mla_sparse[t={tokens}]", "flaggems", build, gems)
    ov = attempt(f"flash_mla_sparse[t={tokens}]", "vllm", build, vllm_)
    if og is not None and ov is not None:
        a = og[0] if isinstance(og, (tuple, list)) else og
        b = ov[0] if isinstance(ov, (tuple, list)) else ov
        rec(op=f"flash_mla_sparse[t={tokens}]", side="diff", max_abs_diff=diff(a, b))


# ------------------------------------------------------------------- flash_mla
def mla_decode(bs):
    hq, hkv, dqk, dv = HEADS_PER_RANK, 1, 576, 512
    block, seq = 64, 4096
    pad = seq

    def build():
        g = torch.Generator().manual_seed(17)
        q = torch.randn(bs, 1, hq, dqk, generator=g).to(torch.bfloat16).to(d)
        nblocks = pad // block
        blocked_k = torch.randn(nblocks * bs, block, hkv, dqk,
                                generator=g).to(torch.bfloat16).to(d)
        bt = torch.arange(nblocks * bs, dtype=torch.int32,
                          device=d).reshape(bs, nblocks)
        cs = torch.full((bs,), seq, dtype=torch.int32, device=d)
        return q, bt, blocked_k, cs

    import flag_gems

    def gems(a):
        q, bt, bk, cs = a
        return flag_gems.flash_mla(q, bt, bk, pad, block, bs, 1, cs,
                                   hq, hkv, dqk, dv, True)

    from vllm.v1.attention.ops.flashmla import (flash_mla_with_kvcache,
                                                get_mla_metadata)

    def vllm_(a):
        q, bt, bk, cs = a
        meta, splits = get_mla_metadata(cs, 1 * hq // hkv, hkv)
        return flash_mla_with_kvcache(q, bk, bt, cs, dv,
                                      tile_scheduler_metadata=meta,
                                      num_splits=splits, causal=True)

    og = attempt(f"flash_mla_decode[bs={bs}]", "flaggems", build, gems)
    ov = attempt(f"flash_mla_decode[bs={bs}]", "vllm", build, vllm_)
    if og is not None and ov is not None:
        a = og[0] if isinstance(og, (tuple, list)) else og
        b = ov[0] if isinstance(ov, (tuple, list)) else ov
        rec(op=f"flash_mla_decode[bs={bs}]", side="diff", max_abs_diff=diff(a, b))


# --------------------------------------------------------- cutlass_scaled_mm
def scaled_mm(M):
    K, N = H, 2048

    def build():
        g = torch.Generator().manual_seed(19)
        a = torch.randint(-8, 8, (M, K), generator=g, dtype=torch.int8).to(d)
        b = torch.randint(-8, 8, (K, N), generator=g, dtype=torch.int8).to(d)
        b = b.t().contiguous().t()  # column-major, as the op asserts
        asc = torch.rand(M, generator=g, dtype=torch.float32).to(d)
        bsc = torch.rand(N, generator=g, dtype=torch.float32).to(d)
        return a, b, asc, bsc

    import flag_gems

    def gems(x):
        a, b, asc, bsc = x
        c = torch.empty(M, N, dtype=torch.bfloat16, device=d)
        flag_gems.cutlass_scaled_mm(c, a, b, asc, bsc, None)
        return c

    def vllm_(x):
        a, b, asc, bsc = x
        return torch.ops._C.cutlass_scaled_mm and None or None

    from vllm import _custom_ops as vops

    def vllm_real(x):
        a, b, asc, bsc = x
        return vops.cutlass_scaled_mm(a, b, asc, bsc, torch.bfloat16, None)

    og = attempt(f"cutlass_scaled_mm_int8[M={M}]", "flaggems", build, gems)
    ov = attempt(f"cutlass_scaled_mm_int8[M={M}]", "vllm", build, vllm_real)
    if og is not None and ov is not None:
        rec(op=f"cutlass_scaled_mm_int8[M={M}]", side="diff",
            max_abs_diff=diff(og, ov))


CASES = {
    "hc_head": lambda: [hc_head(t) for t in (64, 256, 1024, 4096)],
    "group_mm": lambda: [group_mm(t) for t in (192, 1024, 4096)],
    "mla_sparse": lambda: [mla_sparse(t) for t in (64, 1024)],
    "mla_decode": lambda: [mla_decode(b) for b in (64, 256)],
    "scaled_mm": lambda: [scaled_mm(m) for m in (512, 4096)],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=sorted(CASES))
    a = ap.parse_args()
    print(f"===== {a.case} =====")
    CASES[a.case]()
