"""Which kernel actually runs for each candidate op, with and without FlagGems.

Evidence is the kernel symbol from the torch profiler, which is the same
evidence the coverage capture used. A Gems dispatch shows a Triton kernel; a
native dispatch shows at::native::* or a vendor GEMM symbol.

Run twice by the caller: once with GEMS=1 and once with GEMS=0, then diff the
kernel names per op. Same process cannot do both -- registration is global and
not cleanly reversible.
"""
import json
import os
import sys

GEMS = os.environ.get("GEMS", "1") == "1"
os.environ.setdefault("FLAGGEMS_ENABLE_OPLIST_PATH", f"/tmp/gems_disp_oplist_{GEMS}.txt")

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

if GEMS:
    import flag_gems

    flag_gems.enable(record=True, once=True, path=f"/tmp/gems_disp_{GEMS}.log")

dtype = torch.bfloat16
d = "cuda"
torch.manual_seed(0)

# Shapes taken from the traced model where knowable; mm uses a DSv4-Pro-like
# projection shape so the comparison is representative rather than toy.
M, K, N = 4096, 7168, 2048
a = torch.randn(M, K, dtype=dtype, device=d)
b = torch.randn(K, N, dtype=dtype, device=d)
big = torch.randn(M, N, dtype=dtype, device=d)
bias = torch.randn(N, dtype=dtype, device=d)

CASES = {
    "mm": lambda: torch.mm(a, b),
    "addmm": lambda: torch.addmm(bias, a, b),
    "bmm": lambda: torch.bmm(a.unsqueeze(0), b.unsqueeze(0)),
    "clamp": lambda: torch.clamp(big, -1.0, 1.0),
    "clamp_": lambda: big.clone().clamp_(-1.0, 1.0),
    "copy_": lambda: torch.empty_like(big).copy_(big),
}

out = {"gems_enabled": GEMS, "cases": {}}

for name, fn in CASES.items():
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
        kers = {}
        for e in p.key_averages():
            # device-side entries only
            if str(getattr(e, "device_type", "")).endswith("CUDA") or getattr(
                e, "self_device_time_total", 0
            ):
                if getattr(e, "self_device_time_total", 0) > 0:
                    kers[e.key] = round(e.self_device_time_total / 5, 1)
        out["cases"][name] = {
            "kernels": dict(sorted(kers.items(), key=lambda kv: -kv[1])[:6])
        }
    except Exception as e:
        out["cases"][name] = {"error": repr(e)[:200]}

print(json.dumps(out, indent=2, ensure_ascii=False))
with open(f"/tmp/gems_disp_{GEMS}.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
