"""Device-side kernel-time A/B: FlagGems vs vendor-native, per operator.

Replaces the CUDA-event version for small operators. Wrapping one call in CUDA
events measured host launch + Triton dispatch overhead, which for an elementwise
kernel dwarfs the kernel itself: clamp cost the same ~105us at int32[64,256] and
int32[64,16384], a 64x data difference. That is a measurement artifact.

Here the profiler reports self_device_time_total summed over N iterations and
divided by N, which is the actual kernel occupancy, and the kernel symbol is
recorded so the dispatch target is verified rather than assumed.

Both numbers matter and are reported separately:
  kernel_us  -- device time, what the operator costs the GPU
  wall_us    -- host-side per-call cost incl. dispatch, which is what an eager
                (non-graph) serve loop actually pays

Shapes: `traced` from coverage/spec-run-INCOMPLETE-graph-1024x1024/
kernel_shape_dtype.csv (operator_shape_matched rows); `repr` where the trace has
no shape because the op ran inside a CUDA-graph replay.
"""
import json
import os
import re

GEMS = os.environ.get("GEMS", "1") == "1"
TAG = "gems" if GEMS else "native"
os.environ.setdefault("FLAGGEMS_ENABLE_OPLIST_PATH", f"/tmp/gb2_oplist_{TAG}.txt")

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

REG = {}
if GEMS:
    import flag_gems

    flag_gems.enable(record=True, once=True, path=f"/tmp/gb2_{TAG}.log")
    reg = flag_gems.current_work_registrar
    REG = {
        a: len(getattr(reg, a))
        for a in dir(reg)
        if not a.startswith("__")
        and isinstance(getattr(reg, a, None), (set, list, tuple, dict))
    }

d = "cuda"
N_ITERS = 50
torch.manual_seed(0)

CASES = []


def case(op, note, shape_desc, fn):
    CASES.append((op, note, shape_desc, fn))


for n in (256, 8192, 16384):
    t = torch.randint(-1000, 1000, (64, n), dtype=torch.int32, device=d)
    case("clamp", "traced", f"int32[64,{n}]", lambda t=t: torch.clamp(t, 0, 500))
for n in (8192, 16384):
    t = torch.randint(-1000, 1000, (64, n), dtype=torch.int32, device=d)
    tc = t.clone()
    case("clamp_", "traced", f"int32[64,{n}]", lambda tc=tc: tc.clamp_(0, 500))

tb = torch.randn(4096, 2048, dtype=torch.bfloat16, device=d)
case("clamp", "repr", "bf16[4096,2048]", lambda: torch.clamp(tb, -1.0, 1.0))

src = torch.randn(64, 16, 8080, dtype=torch.bfloat16, device=d)
dst = torch.empty_like(src)
case("copy_", "traced", "bf16[64,16,8080]", lambda: dst.copy_(src))
si = torch.randint(0, 1000, (64,), dtype=torch.int32, device=d)
dl = torch.empty(64, dtype=torch.int64, device=d)
case("copy_", "traced", "int32[64]->int64", lambda: dl.copy_(si))

H = 7168
for M, N in ((64, 2048), (1024, 2048), (4096, 2048), (4096, 7168)):
    a = torch.randn(M, H, dtype=torch.bfloat16, device=d)
    b = torch.randn(H, N, dtype=torch.bfloat16, device=d)
    case("mm", "repr", f"bf16[{M},{H}]x[{H},{N}]", lambda a=a, b=b: torch.mm(a, b))

a4 = torch.randn(4096, H, dtype=torch.bfloat16, device=d)
b4 = torch.randn(H, 2048, dtype=torch.bfloat16, device=d)
bias = torch.randn(2048, dtype=torch.bfloat16, device=d)
case("addmm", "repr", f"bf16[4096,{H}]x[{H},2048]+bias", lambda: torch.addmm(bias, a4, b4))
case("bmm", "repr", f"bf16[1,4096,{H}]x[1,{H},2048]",
     lambda: torch.bmm(a4.unsqueeze(0), b4.unsqueeze(0)))

ATEN = re.compile(r"^aten::|^ProfilerStep|^cuda|^Memcpy|^Memset")

out = {"tag": TAG, "gems": GEMS, "registrar": REG, "rows": []}
store = {}

for op, note, shape_desc, fn in CASES:
    key = f"{op}|{shape_desc}"
    rec = {"op": op, "note": note, "shape": shape_desc}
    try:
        for _ in range(15):
            fn()
        torch.cuda.synchronize()

        # wall: host-side per-call
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(N_ITERS):
            fn()
        e.record()
        torch.cuda.synchronize()
        rec["wall_us"] = round(s.elapsed_time(e) * 1000.0 / N_ITERS, 2)

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            for _ in range(N_ITERS):
                fn()
            torch.cuda.synchronize()

        tot, kers = 0.0, {}
        for ev in p.key_averages():
            sd = getattr(ev, "self_device_time_total", 0) or 0
            if sd > 0 and not ATEN.match(ev.key):
                kers[ev.key[:110]] = round(sd / N_ITERS, 2)
                tot += sd
        rec["kernel_us"] = round(tot / N_ITERS, 2)
        rec["kernels"] = dict(sorted(kers.items(), key=lambda kv: -kv[1])[:4])

        r = fn()
        if isinstance(r, torch.Tensor):
            store[key] = r.detach().float().cpu()
    except Exception as ex:
        rec["error"] = repr(ex)[:180]
    out["rows"].append(rec)
    print(f"  {key:42s} kernel={rec.get('kernel_us','ERR'):>9} wall={rec.get('wall_us','ERR'):>9}"
          f"  {list(rec.get('kernels',{}))[:1]}")

with open(f"/tmp/gb2_{TAG}.json", "w") as f:
    json.dump(out, f, indent=2)
torch.save(store, f"/tmp/gb2_{TAG}.pt")
print(f"\nregistrar={REG}\n-> /tmp/gb2_{TAG}.json")
