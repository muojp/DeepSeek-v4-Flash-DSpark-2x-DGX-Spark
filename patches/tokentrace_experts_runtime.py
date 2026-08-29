# [tokentrace-hotfix] runtime module — installed by
# patches/hotfix-dsv4-tokentrace-experts.py as
# vllm/v1/worker/gpu/tokentrace_experts.py (V2 GPU model runner).
"""Per-step expert-routing recorder for the V2 GPU model runner.

Why not ``--enable-return-routed-experts``: on this build DSpark speculative
decoding exists only in the V2 model runner, and V2 rejects the upstream
routed-experts capture. This module reuses the *router-side* half of that
feature (``BaseRouter.set_capture_fn`` — the copy into a fixed device buffer
is part of the model forward, so it is captured into the CUDA graph and
replays every step) and replaces the scheduler/API half with an
append-only per-node log written by the worker itself.

Files (in ``DSPARK_TOKENTRACE_DIR``, default ``/cache/huggingface/tokentrace``
= host ``~/.cache/huggingface/tokentrace``):

  experts-<host>-r<rank>-<pid>-<utc>.idx.jsonl   one JSON line per engine step
  experts-<host>-r<rank>-<pid>-<utc>.u8          uint8 [num_tokens, L, K] per step, appended

Index line: ``{"step":n,"t":wall,"n":num_tokens,"off":byte_offset,"len":bytes,
"req":[ids],"sched":[tokens per req],"pos":[first position per req],
"draft":[draft tokens per req]|null,"sampled":[per req],"rejected":[per req]}``.
Token rows are in the same order as the model saw them (request order of
``InputBatch.req_ids`` with ``sched`` tokens each), so row ``pos[i]+j`` of
request ``i`` is the token at sequence position ``pos[i]+j``.

Overhead per step: one device int32→uint8 conversion of ``num_tokens×L×K``
elements, one non-blocking D2H into pinned memory (≈ 258 B per token), two
tiny D2H for the sampler counts, and a file append when the copy has
landed (checked with a CUDA event; never blocks the main stream). Any
exception inside the recorder disables it for the rest of the process and
logs once — it can never take the engine down.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import time
from datetime import datetime, timezone

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENV_ENABLE = "DSPARK_TOKENTRACE_EXPERTS"
ENV_DIR = "DSPARK_TOKENTRACE_DIR"
DEFAULT_DIR = "/cache/huggingface/tokentrace"
RING = 4


def _num_layers(hf_config) -> int:
    return int(getattr(hf_config, "num_hidden_layers"))


def _topk(hf_config) -> int:
    for k in ("num_experts_per_tok", "moe_topk", "num_experts_per_token", "moe_k"):
        v = getattr(hf_config, k, None)
        if v is not None:
            return int(v)
    raise ValueError("cannot determine top-k experts per token from hf_config")


class ExpertTraceRecorder:
    def __init__(self, *, num_layers: int, topk: int, max_tokens: int, device, out_dir: str,
                 rank: int, host: str | None = None, meta: dict | None = None):
        self.L, self.K, self.max_tokens = num_layers, topk, max_tokens
        self.device = device
        self.buf = torch.zeros((max_tokens, num_layers, topk), dtype=torch.int32, device=device)
        self.u8 = torch.zeros((max_tokens, num_layers, topk), dtype=torch.uint8, device=device)
        self.ring = [torch.empty((max_tokens, num_layers, topk), dtype=torch.uint8, pin_memory=True) for _ in range(RING)]
        self.ring_ns = [torch.empty((max_tokens,), dtype=torch.int32, pin_memory=True) for _ in range(RING)]
        self.ring_nr = [torch.empty((max_tokens,), dtype=torch.int32, pin_memory=True) for _ in range(RING)]
        self.events = [torch.cuda.Event() for _ in range(RING)]
        self.pending: list[dict] = []  # metas whose ring slot is still in flight
        self.slot = 0
        self.step = 0
        self.disabled = False
        self.bytes_written = 0
        self._last_flush = time.monotonic()
        host = host or platform.node().split(".")[0] or socket.gethostname()
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = os.path.join(out_dir, f"experts-{host}-r{rank}-{os.getpid()}-{stamp}")
        self.idx_path, self.data_path = base + ".idx.jsonl", base + ".u8"
        self.idx = open(self.idx_path, "a", encoding="utf-8")
        self.data = open(self.data_path, "ab")
        self.idx.write(json.dumps({"type": "meta", "t": time.time(), "host": host, "rank": rank, "pid": os.getpid(),
                                   "layers": num_layers, "topk": topk, "max_tokens": max_tokens, "dtype": "u8",
                                   "layout": "[n, layers, topk] C-order", **(meta or {})}) + "\n")
        self.idx.flush()

    # ── model side (inside the CUDA graph) ───────────────────────────
    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        if layer_id >= self.L or topk_ids.ndim != 2:
            return
        n = min(topk_ids.shape[0], self.max_tokens)
        self.buf[:n, layer_id, :].copy_(topk_ids[:n, : self.K])

    def bind(self, layers) -> int:
        """``layers``: iterable of (layer_id, router) — routers expose
        ``set_capture_fn``. Returns how many were bound."""
        bound = 0
        for layer_id, router in layers:
            lid = int(layer_id)

            def _fn(topk_ids, _lid=lid, _self=self):
                _self.capture(_lid, topk_ids)

            router.set_capture_fn(_fn)
            bound += 1
        return bound

    # ── runner side (after sampling, once per step) ──────────────────
    def record(self, input_batch, num_sampled: torch.Tensor, num_rejected: torch.Tensor) -> None:
        if self.disabled:
            return
        try:
            n = int(input_batch.num_tokens)
            if n <= 0:
                return
            n = min(n, self.max_tokens)
            s = self.slot
            # slot reuse: make sure its previous D2H has been drained
            if any(m["slot"] == s for m in self.pending):
                self._drain(force=True)
            stream = torch.cuda.current_stream(self.device)
            with torch.cuda.stream(stream):
                self.u8[:n].copy_(self.buf[:n])  # int32 → uint8 on device (experts < 256)
                self.ring[s][:n].copy_(self.u8[:n], non_blocking=True)
                nreq = int(input_batch.num_reqs)
                self.ring_ns[s][:nreq].copy_(num_sampled[:nreq].to(torch.int32), non_blocking=True)
                self.ring_nr[s][:nreq].copy_(num_rejected[:nreq].to(torch.int32), non_blocking=True)
                self.events[s].record(stream)
            draft = getattr(input_batch, "num_draft_tokens_per_req", None)
            self.pending.append({
                "slot": s, "step": self.step, "t": time.time(), "n": n, "nreq": nreq,
                "req": list(input_batch.req_ids),
                "sched": [int(x) for x in input_batch.num_scheduled_tokens[:nreq]],
                "pos": [int(x) for x in input_batch.num_computed_tokens_np[:nreq]],
                "draft": None if draft is None else [int(x) for x in draft[:nreq]],
            })
            self.step += 1
            self.slot = (s + 1) % RING
            self._drain(force=False)
        except Exception as e:  # noqa: BLE001 — never take the engine down
            self.disabled = True
            logger.warning("[tokentrace] expert recorder disabled after error: %r", e)

    def _drain(self, force: bool) -> None:
        while self.pending:
            m = self.pending[0]
            ev = self.events[m["slot"]]
            if force:
                ev.synchronize()
            elif not ev.query():
                break
            self.pending.pop(0)
            self._write(m)

    def _write(self, m: dict) -> None:
        s, n, nreq = m["slot"], m["n"], m["nreq"]
        blob = self.ring[s][:n].numpy().tobytes()
        off = self.data.tell()
        self.data.write(blob)
        self.bytes_written += len(blob)
        rec = {"step": m["step"], "t": m["t"], "n": n, "off": off, "len": len(blob), "req": m["req"],
               "sched": m["sched"], "pos": m["pos"], "draft": m["draft"],
               "sampled": self.ring_ns[s][:nreq].tolist(), "rejected": self.ring_nr[s][:nreq].tolist()}
        self.idx.write(json.dumps(rec, separators=(",", ":")) + "\n")
        now = time.monotonic()
        if m["step"] % 50 == 0 or now - self._last_flush >= 2.0:
            self.idx.flush()
            self.data.flush()
            self._last_flush = now

    def flush(self) -> None:
        try:
            self._drain(force=True)
            self.idx.flush()
            self.data.flush()
            os.fsync(self.idx.fileno())
            os.fsync(self.data.fileno())
        except Exception as e:  # noqa: BLE001
            logger.warning("[tokentrace] flush failed: %r", e)

    def close(self) -> None:
        self.flush()
        try:
            self.idx.close()
            self.data.close()
        except Exception:  # noqa: BLE001
            pass


def collect_moe_layers(static_forward_context) -> list[tuple[int, object]]:
    """(layer_id, router) for every MoERunner whose router supports capture."""
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    out = []
    for module in static_forward_context.values():
        if isinstance(module, MoERunner) and isinstance(getattr(module, "router", None), BaseRouter):
            try:
                out.append((int(module.layer_id), module.router))
            except Exception:  # noqa: BLE001 — unusual prefix, skip
                continue
    return out


def maybe_init_expert_trace(runner) -> ExpertTraceRecorder | None:
    """Called from GPUModelRunner.initialize_kv_cache (before CUDA-graph
    capture). Returns None unless DSPARK_TOKENTRACE_EXPERTS=1."""
    if os.environ.get(ENV_ENABLE, "0") != "1":
        return None
    try:
        vc = runner.vllm_config
        hf = vc.model_config.hf_text_config
        rank = int(getattr(vc.parallel_config, "rank", 0) or 0)
        rec = ExpertTraceRecorder(
            num_layers=_num_layers(hf), topk=_topk(hf),
            max_tokens=int(vc.scheduler_config.max_num_batched_tokens),
            device=runner.device, out_dir=os.environ.get(ENV_DIR, DEFAULT_DIR), rank=rank,
            meta={"model": vc.model_config.model, "tp": vc.parallel_config.tensor_parallel_size,
                  "spec": getattr(vc.speculative_config, "method", None) if vc.speculative_config else None},
        )
        layers = collect_moe_layers(vc.compilation_config.static_forward_context)
        bound = rec.bind(layers)
        logger.info("[tokentrace] expert recorder on: %d MoE layers bound (ids %s..%s), files %s",
                    bound, min((l for l, _ in layers), default=None), max((l for l, _ in layers), default=None), rec.idx_path)
        if bound == 0:
            rec.close()
            return None
        return rec
    except Exception as e:  # noqa: BLE001
        logger.warning("[tokentrace] expert recorder NOT enabled: %r", e)
        return None
