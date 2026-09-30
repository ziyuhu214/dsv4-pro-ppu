"""Correctness A/B with identical inputs, plus a registrar diff vs the serve process.

Two fixes over the previous attempt:
  1. Inputs are generated on CPU under a fixed seed and copied to device. With
     FlagGems enabled, torch.randn/randint are themselves dispatched to Gems RNG
     kernels, so device-side generation produced different data in the two runs
     and made every float comparison meaningless.
  2. Dumps this process's registrar key set so it can be diffed against the
     854 keys the serve worker recorded (worker_audit_rank0_pid805387.json).
"""
import json
import os

GEMS = os.environ.get("GEMS", "1") == "1"
TAG = "gems" if GEMS else "native"
os.environ.setdefault("FLAGGEMS_ENABLE_OPLIST_PATH", f"/tmp/gv_oplist_{TAG}.txt")

import torch  # noqa: E402

keys = []
if GEMS:
    import flag_gems

    flag_gems.enable(record=True, once=True, path=f"/tmp/gv_{TAG}.log")
    reg = flag_gems.current_work_registrar
    ak = getattr(reg, "all_keys", None)
    keys = sorted(ak) if ak else []

d = "cuda"
g = torch.Generator().manual_seed(1234)  # CPU generator: identical in both runs


def cpu_randn(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, generator=g, dtype=torch.float32).to(dtype).to(d)


def cpu_randint(lo, hi, shape, dtype=torch.int32):
    return torch.randint(lo, hi, shape, generator=g, dtype=dtype).to(d)


H = 7168
res = {"tag": TAG, "registrar_keys": len(keys), "out": {}}
store = {}


def run(key, fn):
    try:
        r = fn()
        torch.cuda.synchronize()
        store[key] = r.detach().float().cpu()
        res["out"][key] = "ok"
    except Exception as e:
        res["out"][key] = repr(e)[:160]


for n in (256, 8192, 16384):
    t = cpu_randint(-1000, 1000, (64, n))
    run(f"clamp|int32[64,{n}]", lambda t=t: torch.clamp(t, 0, 500))
for n in (8192, 16384):
    t = cpu_randint(-1000, 1000, (64, n))
    run(f"clamp_|int32[64,{n}]", lambda t=t: t.clone().clamp_(0, 500))

tb = cpu_randn(4096, 2048)
run("clamp|bf16[4096,2048]", lambda: torch.clamp(tb, -1.0, 1.0))

src = cpu_randn(64, 16, 8080)
run("copy_|bf16[64,16,8080]", lambda: torch.empty_like(src).copy_(src))
si = cpu_randint(0, 1000, (64,))
run("copy_|int32[64]->int64",
    lambda: torch.empty(64, dtype=torch.int64, device=d).copy_(si))

for M, N in ((64, 2048), (1024, 2048), (4096, 2048)):
    a = cpu_randn(M, H)
    b = cpu_randn(H, N)
    run(f"mm|bf16[{M},{H}]x[{H},{N}]", lambda a=a, b=b: torch.mm(a, b))

a4 = cpu_randn(4096, H)
b4 = cpu_randn(H, 2048)
bias = cpu_randn(2048)
run("addmm|bf16[4096,7168]x[7168,2048]+bias", lambda: torch.addmm(bias, a4, b4))
run("bmm|bf16[1,4096,7168]x[1,7168,2048]",
    lambda: torch.bmm(a4.unsqueeze(0), b4.unsqueeze(0)))

torch.save(store, f"/tmp/gv_{TAG}.pt")
with open(f"/tmp/gv_{TAG}.json", "w") as f:
    json.dump({**res, "keys": keys}, f, indent=2)
print(json.dumps(res, indent=2)[:1500])
print("registrar keys:", len(keys))
