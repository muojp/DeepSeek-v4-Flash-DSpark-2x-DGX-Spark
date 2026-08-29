"""tokentrace.probe — streaming client against the vLLM stub (pytest)."""
import base64
import json
import struct
from pathlib import Path

from conftest import free_port
from tokentrace import control
from tokentrace.probe import DEFAULT_PROMPT, build_parser, main, run_routed_experts, run_stream
from tokentrace.writer import TraceWriter, iter_jsonl


def _probe_records(tmp_path):
    files = sorted((tmp_path / "trace" / "testhost").glob("probe-*.jsonl"))
    assert files
    return [r for f in files for r in iter_jsonl(f)]


def test_stream_records_chunks_usage_and_arms_samplers(tmp_path, vllm_stub):
    url, state = vllm_stub
    state["words"] = ["The", " quick", " brown", " fox", " jumps", " over"]
    arms, marks = [], []
    srv = control.ControlServer("127.0.0.1", 0, on_arm=lambda s, who: arms.append(s), on_mark=lambda l, n, who: marks.append(l),
                                on_stat=lambda: {})
    port = srv.sock.getsockname()[1]
    srv.start()
    try:
        rc = main(["--url", url, "--dir", str(tmp_path / "trace"), "--host", "testhost", "--max-tokens", "6",
                   "--thinking", "off", "--arm", f"127.0.0.1:{port}", "--arm-secs", "42", "--label", "t", "--save-text"])
    finally:
        srv.stop()
    assert rc == 0
    assert arms == [42.0] and marks == ["probe_start", "probe_end"]
    recs = _probe_records(tmp_path)
    req = [r for r in recs if r["type"] == "request"][0]
    chunks = [r for r in recs if r["type"] == "chunk"]
    resp = [r for r in recs if r["type"] == "response"][0]
    assert req["max_tokens"] == 6 and req["thinking"] == "off" and req["prompt_chars"] == len(DEFAULT_PROMPT)
    assert len(chunks) == 6 and [c["i"] for c in chunks] == list(range(6))
    assert chunks[1]["chars"] == len(" quick") and chunks[-1]["finish"] == "length"
    assert all(chunks[i]["t"] <= chunks[i + 1]["t"] for i in range(5))
    assert resp["chunks"] == 6 and resp["usage"]["completion_tokens"] == 6 and resp["finish"] == "length"
    assert resp["ttfc_s"] is not None and resp["stream_s"] > 0 and resp["decode_tok_s"] > 0
    assert resp["text_chars"] == len("The quick brown fox jumps over")
    txt = list((tmp_path / "trace" / "testhost").glob("tt-*.txt"))
    assert txt and txt[0].read_text() == "The quick brown fox jumps over"
    sent = state["requests"][0]
    assert sent["stream"] is True and sent["chat_template_kwargs"] == {"thinking": False} and sent["max_tokens"] == 6


def test_thinking_effort_and_min_tokens_are_forwarded(tmp_path, vllm_stub):
    url, state = vllm_stub
    a = build_parser().parse_args(["--url", url, "--thinking", "high", "--min-tokens", "3", "--ignore-eos", "--max-tokens", "5"])
    w = TraceWriter(tmp_path, "probe", "h", min_free_start=0)
    r = run_stream(a, w, "req-1", "hi")
    w.close()
    assert not r.get("error") and r["chunks"] == 5
    sent = state["requests"][-1]
    assert sent["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": "high"}
    assert sent["min_tokens"] == 3 and sent["ignore_eos"] is True


def test_routed_experts_replay_saves_npy_when_present(tmp_path, vllm_stub):
    url, state = vllm_stub
    header = "{'descr': '|u1', 'fortran_order': False, 'shape': (2, 3, 2), }"
    header += " " * (64 - (10 + len(header) + 1) % 64) + "\n"
    npy = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode() + bytes(range(12))
    state["routed_b64"] = base64.b64encode(npy).decode()
    a = build_parser().parse_args(["--url", url, "--max-tokens", "5"])
    w = TraceWriter(tmp_path, "probe", "h", min_free_start=0)
    rec = run_routed_experts(a, w, "req-2", "hi")
    w.close()
    assert rec["present"] and rec["bytes"] == len(npy)
    assert (tmp_path / rec["file"]).read_bytes() == npy
    state["routed_b64"] = None
    w = TraceWriter(tmp_path, "probe", "h", min_free_start=0)
    rec = run_routed_experts(a, w, "req-3", "hi")
    w.close()
    assert rec["present"] is False and "file" not in rec


def test_server_error_is_recorded_and_exit_code_nonzero(tmp_path, vllm_stub):
    url, state = vllm_stub
    state["drop"] = True
    rc = main(["--url", url, "--dir", str(tmp_path / "trace"), "--host", "testhost", "--max-tokens", "3", "--timeout", "2"])
    assert rc == 1
    resp = [r for r in _probe_records(tmp_path) if r["type"] == "response"][0]
    assert resp["error"] and resp["chunks"] == 0


def test_repeat_and_prompt_file(tmp_path, vllm_stub):
    url, state = vllm_stub
    pf = tmp_path / "p.txt"
    pf.write_text("custom prompt")
    rc = main(["--url", url, "--dir", str(tmp_path / "trace"), "--host", "testhost", "--max-tokens", "2", "--repeat", "2",
               "--prompt-file", str(pf)])
    assert rc == 0
    recs = _probe_records(tmp_path)
    reqs = [r for r in recs if r["type"] == "request"]
    assert len(reqs) == 2 and reqs[0]["req"] != reqs[1]["req"] and reqs[0]["prompt_chars"] == len("custom prompt")
    assert state["requests"][-1]["messages"][0]["content"] == "custom prompt"


def test_unreachable_arm_targets_do_not_block(tmp_path, vllm_stub):
    url, _ = vllm_stub
    dead = free_port()
    rc = main(["--url", url, "--dir", str(tmp_path / "trace"), "--host", "testhost", "--max-tokens", "2",
               "--arm", f"127.0.0.1:{dead}"])
    assert rc == 0  # arming is best-effort
