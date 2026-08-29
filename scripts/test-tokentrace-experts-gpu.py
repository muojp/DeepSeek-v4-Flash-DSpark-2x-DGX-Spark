#!/usr/bin/env python3
"""GPU self-test for patches/tokentrace_experts_runtime.py.

Runs inside the dspark vLLM image (needs torch + CUDA, a few MB of GPU
memory, no model). Exercises exactly the mechanism the hotfix relies on:

  1. routers get a capture_fn; a *captured CUDA graph* calls it
     (as the real MoE router forward does inside the model graph);
  2. graph replay with new inputs updates the fixed device buffer;
  3. record() issues the non-blocking D2H + event, drains to the index/data
     files, and the files decode back to the exact routing that was replayed;
  4. ring-slot reuse, partial batches, tokens beyond max_tokens, an exception
     inside record() disabling the recorder without raising.

Usage (throwaway container, GPU shared with the running server):
  docker run --rm --gpus all -v $PWD/patches:/p:ro -v $PWD/scripts:/s:ro \
     ghcr.io/anemll/dspark-vllm-gx10:0.1.1 python3 /s/test-tokentrace-experts-gpu.py
"""
import importlib.util
import json
import os
import sys
import tempfile
import types

import numpy as np
import torch

SRC = os.environ.get("TT_RUNTIME", "/p/tokentrace_experts_runtime.py")


def load_runtime():
    # the module imports vllm.logger; provide a stub when vllm is absent
    try:
        import vllm.logger  # noqa: F401
    except Exception:
        stub = types.ModuleType("vllm"); lg = types.ModuleType("vllm.logger")
        import logging
        lg.init_logger = lambda name: logging.getLogger(name)
        stub.logger = lg
        sys.modules["vllm"] = stub; sys.modules["vllm.logger"] = lg
    spec = importlib.util.spec_from_file_location("tokentrace_experts", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeRouter:
    def __init__(self):
        self.capture_fn = None

    def set_capture_fn(self, fn):
        self.capture_fn = fn

    def forward(self, topk_ids):  # what BaseRouter.select_experts does at the end
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)


def fake_batch(req_ids, sched, pos, draft=None):
    n = int(sum(sched))
    return types.SimpleNamespace(req_ids=list(req_ids), num_reqs=len(req_ids), num_tokens=n,
                                 num_scheduled_tokens=np.array(sched, dtype=np.int32),
                                 num_computed_tokens_np=np.array(pos, dtype=np.int32),
                                 num_draft_tokens_per_req=None if draft is None else np.array(draft, dtype=np.int32))


def main():
    assert torch.cuda.is_available(), "CUDA required"
    dev = torch.device("cuda:0")
    mod = load_runtime()
    L, K, MAXT = 5, 3, 64
    out = tempfile.mkdtemp(prefix="tt-experts-")
    rec = mod.ExpertTraceRecorder(num_layers=L, topk=K, max_tokens=MAXT, device=dev, out_dir=out, rank=0, host="testhost")
    routers = [FakeRouter() for _ in range(L + 1)]  # +1: a drafter layer with id >= L must be ignored
    bound = rec.bind([(i, r) for i, r in enumerate(routers)])
    assert bound == L + 1

    # static inputs the graph reads (like the router's topk_ids tensors)
    N = 7
    static = [torch.zeros((N, K), dtype=torch.int32, device=dev) for _ in range(L + 1)]

    # warm-up on a side stream, then capture (same recipe torch docs use)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i, r in enumerate(routers):
            r.forward(static[i])
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i, r in enumerate(routers):
            r.forward(static[i])

    # --- replay 1: distinct values per (token, layer, k)
    expect1 = np.zeros((N, L, K), dtype=np.uint8)
    for i in range(L + 1):
        vals = (torch.arange(N * K, device=dev, dtype=torch.int32).reshape(N, K) + 10 * i) % 256
        static[i].copy_(vals)
        if i < L:
            expect1[:, i, :] = vals.cpu().numpy()
    g.replay()
    ns = torch.tensor([1, 3], dtype=torch.int32, device=dev)
    nr = torch.tensor([0, 2], dtype=torch.int32, device=dev)
    rec.record(fake_batch(["a", "b"], [3, 4], [100, 7], draft=[0, 5]), ns, nr)

    # --- replay 2: overwrite, fewer tokens scheduled (n=4)
    for i in range(L + 1):
        static[i].fill_(200 + i)
    g.replay()
    rec.record(fake_batch(["c"], [4], [0]), torch.tensor([4], device=dev), torch.tensor([0], device=dev))

    # --- replays 3..8: ring-slot reuse
    for k in range(6):
        for i in range(L + 1):
            static[i].fill_(k)
        g.replay()
        rec.record(fake_batch(["d"], [2], [k]), torch.tensor([1], device=dev), torch.tensor([0], device=dev))

    # --- n beyond max_tokens is clamped, not an error
    rec.record(fake_batch(["e"], [MAXT + 100], [0]), torch.tensor([1], device=dev), torch.tensor([0], device=dev))

    rec.close()

    # ---- verify files
    idx = [json.loads(l) for l in open(rec.idx_path, encoding="utf-8")]
    meta, steps = idx[0], idx[1:]
    assert meta["type"] == "meta" and meta["layers"] == L and meta["topk"] == K
    data = open(rec.data_path, "rb").read()
    assert len(steps) == 9, len(steps)
    st = steps[0]
    assert st["req"] == ["a", "b"] and st["sched"] == [3, 4] and st["pos"] == [100, 7] and st["draft"] == [0, 5]
    assert st["sampled"] == [1, 3] and st["rejected"] == [0, 2]
    arr = np.frombuffer(data[st["off"]: st["off"] + st["len"]], dtype=np.uint8).reshape(st["n"], L, K)
    assert np.array_equal(arr, expect1), (arr[:2], expect1[:2])
    st2 = steps[1]
    arr2 = np.frombuffer(data[st2["off"]: st2["off"] + st2["len"]], dtype=np.uint8).reshape(4, L, K)
    for i in range(L):
        assert (arr2[:, i, :] == 200 + i).all()
    for k in range(6):
        stk = steps[2 + k]
        a = np.frombuffer(data[stk["off"]: stk["off"] + stk["len"]], dtype=np.uint8).reshape(2, L, K)
        assert (a == k).all(), (k, a)
    assert steps[8]["n"] == MAXT
    assert sum(s["len"] for s in steps) == len(data)

    # ---- exception inside record disables, never raises
    rec2 = mod.ExpertTraceRecorder(num_layers=L, topk=K, max_tokens=MAXT, device=dev, out_dir=out, rank=1, host="testhost")
    rec2.record(types.SimpleNamespace(num_tokens=3), None, None)  # missing attrs → exception path
    assert rec2.disabled
    rec2.record(fake_batch(["x"], [1], [0]), ns, nr)  # no-op now
    rec2.close()

    # ---- overhead estimate: 1000 record() calls with n=12 (decode-like), graph replay each
    for i in range(L + 1):
        static[i].fill_(1)
    rec3 = mod.ExpertTraceRecorder(num_layers=43, topk=6, max_tokens=8192, device=dev, out_dir=out, rank=2, host="testhost")
    b = fake_batch(["p", "q"], [6, 6], [0, 0])
    torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(1000):
        rec3.record(b, ns, nr)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 1000
    rec3.close()
    print(f"OK  files={rec.idx_path}  data_bytes={len(data)}  record() decode-size cost ≈ {dt*1e6:.0f} µs/step "
          f"(real model: 43x6 uint8 per token)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
