"""Streaming request probe: timestamps every SSE chunk, arms the samplers on
both nodes, and (when the server exposes it) stores the routed-experts
array of a non-streaming replay of the same prompt."""
from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import time
import urllib.request
import uuid
from pathlib import Path

from . import control
from .writer import TraceWriter

DEFAULT_PROMPT = (
    "You are writing a technical field guide. Explain, in numbered sections with concrete examples, "
    "how a mixture-of-experts transformer routes tokens to experts, why unified memory changes the cost "
    "model of expert switching compared with discrete GPUs, and how speculative decoding with a multi-token "
    "predictor interacts with tensor parallelism across two machines linked by RoCE. Keep going in detail "
    "until you are stopped; do not summarise early."
)


def _post(url: str, payload: dict, timeout: float, stream: bool):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "Accept": "text/event-stream" if stream else "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def run_stream(a, writer: TraceWriter, req_id: str, prompt: str) -> dict:
    payload = {
        "model": a.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": a.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": a.temperature,
    }
    if a.min_tokens:
        payload["min_tokens"] = a.min_tokens
    if a.ignore_eos:
        payload["ignore_eos"] = True
    if a.thinking != "default":
        payload["chat_template_kwargs"] = {"thinking": a.thinking != "off"}
        if a.thinking not in ("off", "on"):
            payload["chat_template_kwargs"]["reasoning_effort"] = a.thinking
    writer.write({"type": "request", "req": req_id, "model": a.model, "prompt_chars": len(prompt),
                  "max_tokens": a.max_tokens, "thinking": a.thinking, "stream": True, "t_send": time.time()})
    t_send = time.monotonic()
    usage = None
    n = 0
    t_first = t_last = None
    text_parts = []
    finish = None
    try:
        r = _post(a.url.rstrip("/") + "/v1/chat/completions", payload, a.timeout, stream=True)
        for raw in r:
            now_m = time.monotonic()
            now_w = time.time()
            if not raw.startswith(b"data:"):
                continue
            body = raw[5:].strip()
            if body == b"[DONE]":
                break
            try:
                obj = json.loads(body)
            except ValueError:
                continue
            if obj.get("usage") and not obj.get("choices"):
                usage = obj["usage"]
                continue
            ch = (obj.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            finish = ch.get("finish_reason") or finish
            if obj.get("usage"):
                usage = obj["usage"]
            if t_first is None:
                t_first = now_m
            t_last = now_m
            text_parts.append(content)
            writer.write({"type": "chunk", "t": now_w, "m": now_m, "req": req_id, "i": n, "bytes": len(raw),
                          "chars": len(content), "reasoning_chars": len(reasoning), "finish": ch.get("finish_reason")})
            n += 1
    except Exception as e:  # noqa: BLE001
        writer.write({"type": "response", "req": req_id, "error": str(e)[:500], "chunks": n})
        return {"error": str(e), "chunks": n}
    t_end = time.monotonic()
    resp = {"type": "response", "req": req_id, "chunks": n, "finish": finish, "usage": usage,
            "t_send_m": t_send, "t_first_chunk_m": t_first, "t_last_chunk_m": t_last, "t_end_m": t_end,
            "ttfc_s": None if t_first is None else round(t_first - t_send, 4),
            "stream_s": None if t_first is None else round(t_last - t_first, 4),
            "text_chars": sum(len(p) for p in text_parts)}
    if usage and t_first is not None and t_last and t_last > t_first:
        ct = usage.get("completion_tokens") or 0
        resp["decode_tok_s"] = round((ct - 1) / (t_last - t_first), 2) if ct > 1 else None
    writer.write(resp)
    if a.save_text:
        (writer.dir / f"{req_id}.txt").write_text("".join(text_parts), encoding="utf-8")
    return resp


def run_routed_experts(a, writer: TraceWriter, req_id: str, prompt: str) -> dict | None:
    """Non-streaming replay; stores the raw .npy bytes when the server has
    --enable-return-routed-experts. Returns None when the field is absent."""
    payload = {"model": a.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": a.max_tokens,
               "temperature": a.temperature, "stream": False}
    if a.thinking != "default":
        payload["chat_template_kwargs"] = {"thinking": a.thinking != "off"}
    t0 = time.time()
    try:
        with _post(a.url.rstrip("/") + "/v1/chat/completions", payload, a.timeout, stream=False) as r:
            obj = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        writer.write({"type": "routed_experts", "req": req_id, "error": str(e)[:500]})
        return None
    ch = (obj.get("choices") or [{}])[0]
    b64 = ch.get("routed_experts")
    rec = {"type": "routed_experts", "req": req_id, "t_send": t0, "present": bool(b64), "usage": obj.get("usage")}
    if b64:
        raw = base64.b64decode(b64)
        fn = f"{req_id}-experts.npy"
        (writer.dir / fn).write_bytes(raw)
        rec["file"] = fn
        rec["bytes"] = len(raw)
        rec["header"] = raw[:128].decode("latin-1")
    writer.write(rec)
    return rec


def build_parser(p: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    p = p or argparse.ArgumentParser(prog="tokentrace probe")
    p.add_argument("--url", default="http://127.0.0.1:8888")
    p.add_argument("--model", default="deepseek-v4-flash-0731")
    p.add_argument("--dir", default="~/tokentrace")
    p.add_argument("--host", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-file", default=None)
    p.add_argument("--max-tokens", type=int, default=1000)
    p.add_argument("--min-tokens", type=int, default=0)
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--thinking", default="off", help="off | on | low | high | max | default")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--arm", default="", help="comma list host[:port] of samplers to arm (both nodes)")
    p.add_argument("--arm-secs", type=float, default=600.0)
    p.add_argument("--routed-experts", action="store_true", help="also replay non-streaming and save routed_experts")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--save-text", action="store_true")
    p.add_argument("--label", default="")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    host = a.host or platform.node().split(".")[0]
    writer = TraceWriter(Path(os.path.expanduser(a.dir)) / host, "probe", host, min_free_start=256 << 20)
    prompt = a.prompt or (Path(a.prompt_file).read_text(encoding="utf-8") if a.prompt_file else DEFAULT_PROMPT)
    targets = []
    for item in [x for x in a.arm.split(",") if x.strip()]:
        h, _, p = item.strip().partition(":")
        targets.append((h, int(p or control.DEFAULT_PORT)))
    results = []
    for i in range(a.repeat):
        req_id = f"tt-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        if targets:
            armed = control.arm(targets, a.arm_secs)
            control.mark(targets, "probe_start", f"{req_id} {a.label}".strip())
            print(f"[probe] armed {armed}", flush=True)
        print(f"[probe] {req_id}: streaming max_tokens={a.max_tokens} thinking={a.thinking}", flush=True)
        r = run_stream(a, writer, req_id, prompt)
        print(f"[probe] {req_id}: chunks={r.get('chunks')} ttfc={r.get('ttfc_s')}s stream={r.get('stream_s')}s "
              f"decode={r.get('decode_tok_s')} tok/s usage={r.get('usage')}", flush=True)
        if a.routed_experts and not r.get("error"):
            control.mark(targets, "probe_replay_start", req_id) if targets else None
            re_rec = run_routed_experts(a, writer, req_id, prompt)
            print(f"[probe] routed_experts: {re_rec}", flush=True)
        if targets:
            control.mark(targets, "probe_end", req_id)
        results.append(r)
    writer.close()
    return 0 if all(not r.get("error") for r in results) else 1
