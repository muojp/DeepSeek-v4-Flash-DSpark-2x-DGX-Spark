"""tokentrace.writer / control / analyze — rotation, free-space guard,
NTP maths, UDP control server, step reconstruction, .npy decoding,
expert-log ingestion, phase summary (pytest)."""
import json
import socket
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tokentrace import analyze, control
from tokentrace.writer import TraceWriter, iter_jsonl

# ── writer ───────────────────────────────────────────────────────────────


def test_writer_stamps_rotates_and_tolerates_truncation(tmp_path):
    days = [datetime(2026, 8, 28, 23, 59, 59, tzinfo=timezone.utc)]
    w = TraceWriter(tmp_path, "trace", "dgx01", flush_s=0, fsync_s=0, min_free_start=0, utcnow=lambda: days[0])
    assert w.write({"type": "fast", "v": [1, 2]})
    p1 = w.path
    days[0] = datetime(2026, 8, 29, 0, 0, 1, tzinfo=timezone.utc)
    assert w.write({"type": "fast", "v": [3, 4]})
    p2 = w.path
    w.close()
    assert p1 != p2
    assert p1.name.endswith("20260828.jsonl") and p2.name.endswith("20260829.jsonl")
    recs = list(iter_jsonl(p1))
    assert len(recs) == 1
    assert {"t", "m", "host", "type"} <= set(recs[0])
    assert recs[0]["host"] == "dgx01"
    with open(p2, "ab") as f:
        f.write(b'{"type":"fast","v":[5,')  # truncated tail is skipped, not fatal
    assert [r["v"] for r in iter_jsonl(p2)] == [[3, 4]]
    assert w.stats()["written"] == 2


def test_writer_refuses_to_start_without_space(tmp_path):
    with pytest.raises(RuntimeError):
        TraceWriter(tmp_path, "trace", "h", min_free_start=1 << 60)


def test_writer_pauses_and_resumes_on_free_space(tmp_path):
    w = TraceWriter(tmp_path, "trace", "h", flush_s=0, fsync_s=0, min_free_start=0, min_free_run=1 << 60)
    w._last_space_check = -1e9
    assert not w.write({"type": "x"})
    assert w.paused and w.dropped == 1
    w.min_free_run = 0
    w._last_space_check = -1e9
    assert w.write({"type": "x"})
    assert not w.paused
    w.close()


def test_writer_rejects_nan(tmp_path):
    w = TraceWriter(tmp_path, "trace", "h", min_free_start=0)
    with pytest.raises(ValueError):
        w.write({"type": "x", "v": float("nan")})
    w.close()


# ── control ──────────────────────────────────────────────────────────────


def test_ntp_offset_math():
    t0 = 100.0
    t1 = t0 + 0.05 + 1.0  # server +1.000 s, 50 ms each way
    t2 = t1 + 0.001
    t3 = t0 + 0.101
    off, rtt = control.ntp_offset(t0, t1, t2, t3)
    assert off == pytest.approx(1.0, abs=1e-6)
    assert rtt == pytest.approx(0.100, abs=1e-6)
    s = control.summarize_offsets([(1.0, 0.1), (1.002, 0.5), (0.999, 0.09), (1.5, 2.0)])
    assert s["n"] == 4 and s["rtt_us"] == 90000
    assert s["offset_us"] / 1e6 == pytest.approx(0.9995, abs=1e-3)
    assert control.summarize_offsets([]) is None


def test_control_server_roundtrip():
    arms, marks = [], []
    srv = control.ControlServer("127.0.0.1", 0, on_arm=lambda s, who: arms.append((s, who)),
                                on_mark=lambda l, n, who: marks.append((l, n)), on_stat=lambda: {"ok": 1})
    port = srv.sock.getsockname()[1]
    srv.start()
    try:
        r = control.measure_offset("127.0.0.1", port, n=4)
        assert r is not None and abs(r["offset_us"]) < 50_000 and r["n"] >= 1
        assert control.arm([("127.0.0.1", port)], 12) == {f"127.0.0.1:{port}": True}
        assert arms[0][0] == 12.0
        assert control.mark([("127.0.0.1", port)], "probe_start", "x")[f"127.0.0.1:{port}"]
        assert marks == [("probe_start", "x")]
        assert control.send("127.0.0.1", port, {"op": "stat"})["ok"] == 1
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(b"\xff\xfe not json", ("127.0.0.1", port))  # garbage must not kill the server
        s.close()
        assert control.send("127.0.0.1", port, {"op": "stat"}) is not None
        assert control.send("127.0.0.1", port, {"op": "bogus"}, timeout=0.2) is None  # no reply for unknown op
    finally:
        srv.stop()
    assert control.send("127.0.0.1", port, {"op": "stat"}, timeout=0.2) is None


def test_control_handler_exception_is_reported_not_fatal():
    def boom(*_):
        raise RuntimeError("nope")
    srv = control.ControlServer("127.0.0.1", 0, on_arm=boom, on_mark=boom, on_stat=lambda: {})
    port = srv.sock.getsockname()[1]
    srv.start()
    try:
        r = control.send("127.0.0.1", port, {"op": "arm", "secs": 1})
        assert r["op"] == "err" and "nope" in r["error"]
    finally:
        srv.stop()


# ── analyze ──────────────────────────────────────────────────────────────


def _node(host, columns, rows, offset=0.0):
    n = analyze.NodeTrace(host)
    n.add({"type": "meta", "t": 0, "columns": {"fast": columns}})
    for t, v in rows:
        n.add({"type": "fast", "t": t, "v": v})
    n.offset_s = offset
    n.finalize()
    return n


def test_interpolation_and_busy_fraction():
    cols = ["ib_tx_bytes", "gpu_util"]
    n = _node("h", cols, [(10.0, [0, 0]), (10.1, [1000, 50]), (10.2, [1000, 50]), (10.3, [3000, 100])])
    assert n.value_at("ib_tx_bytes", 10.05) == pytest.approx(500.0)
    assert n.delta("ib_tx_bytes", 10.0, 10.3) == pytest.approx(3000.0)
    assert n.mean_gauge("gpu_util", 10.05, 10.25) == pytest.approx(50.0)
    assert n.busy_fraction("ib_tx_bytes", 10.0, 10.3, bin_s=0.1) == pytest.approx(2 / 3, abs=0.01)
    assert n.delta("missing", 10.0, 10.3) is None
    assert n.value_at("gpu_util", 5.0) == 0.0 and n.value_at("gpu_util", 50.0) == 100.0  # clamped ends


def test_clock_offset_applied_to_worker():
    cols = ["ib_tx_bytes"]
    head = analyze.NodeTrace("dgx01")
    head.add({"type": "meta", "t": 0, "columns": {"fast": cols}})
    head.add({"type": "fast", "t": 100.0, "v": [0]})
    head.add({"type": "clock", "t": 100.0, "offset_us": 250_000, "rtt_us": 80})
    head.add({"type": "clock", "t": 101.0, "offset_us": 251_000, "rtt_us": 90})
    worker = analyze.NodeTrace("dgx02")
    worker.add({"type": "meta", "t": 0, "columns": {"fast": cols}})
    worker.add({"type": "fast", "t": 100.25, "v": [0]})
    analyze.apply_clock_offsets({"dgx01": head, "dgx02": worker}, "dgx01")
    assert worker.fast_t[0] == pytest.approx(100.25 - 0.2505, abs=1e-6)


def _decode_trace():
    cols = ["ib_tx_bytes", "ib_rx_bytes", "ib_tx_pkts", "gpu_util", "gpu_mem_util", "gpu_power_mw",
            "disk_rd_sectors", "disk_wr_sectors", "pswpin", "pgmajfault", "net_tx_bytes"]
    rows, t, tx = [], 0.0, 0
    while t <= 2.0:
        rows.append((round(t, 3), [tx, tx, tx // 50, 40, 20, 50000, 0, 0, 0, 0, 0]))
        t += 0.01
        tx += 12_800 if t > 1.0 else 500
    head = _node("dgx01", cols, rows)
    head.vllm = [{"t": 0.0, "steps": 1000, "gen_tokens": 5000, "accepted_tokens": 10, "draft_tokens": 20},
                 {"t": 1.0, "steps": 1010, "gen_tokens": 5000, "accepted_tokens": 10, "draft_tokens": 20},
                 {"t": 2.0, "steps": 1024, "gen_tokens": 5030, "accepted_tokens": 40, "draft_tokens": 60}]
    probe = {"r1": {"request": {"t": 0.0, "t_send": 0.0},
                    "chunks": [{"t": 1.0, "i": 0}, {"t": 1.07, "i": 1}, {"t": 1.14, "i": 2}, {"t": 2.0, "i": 3}],
                    "response": {"chunks": 4, "ttfc_s": 1.0, "stream_s": 1.0, "decode_tok_s": 30.0, "usage": {"completion_tokens": 31}}}}
    return head, probe


def test_step_reconstruction_from_chunks():
    head, probe = _decode_trace()
    steps = analyze.build_steps({"dgx01": head}, "dgx01", None, probe)
    assert [s["phase"] for s in steps] == ["prefill", "decode", "decode", "decode"]
    assert steps[0]["dur_ms"] == pytest.approx(1000.0) and steps[0]["engine_steps"] == 10
    assert steps[1]["dur_ms"] == pytest.approx(70.0)
    assert steps[1]["head_ib_tx_bytes"] == pytest.approx(7 * 12_800, abs=13_000)
    assert steps[1]["wait_class"] == "fabric-latency" and steps[3]["wait_class"] == "fabric-latency"
    summ = analyze.summarize(steps, {"dgx01": head}, "dgx01", None, probe)
    assert summ["decode_steps"] == 3 and summ["wait_classes_decode"] == {"fabric-latency": 3}
    assert summ["decode_ib_tx_MB_s"] is not None
    assert "Decode steps" in analyze.render_markdown(summ, steps, None)


def test_step_boundaries_fallback_to_vllm_counter():
    head, _ = _decode_trace()
    steps = analyze.build_steps({"dgx01": head}, "dgx01", None, {})
    assert [s["phase"] for s in steps] == ["unknown", "unknown"]
    assert analyze.step_boundaries_from_vllm(head.vllm) == [(0.0, 1.0, 10), (1.0, 2.0, 14)]


def test_wait_classes_and_paging_flags():
    assert analyze.paging_flags({"head_swap_in": 3, "worker_disk_rd_bytes": 5 << 20}) == ["head-swap-in", "worker-disk-read"]
    assert analyze.paging_flags({"head_disk_rd_bytes": 1000}) == []
    assert analyze.classify_wait({"head_gpu_util": 95, "head_ib_tx_bytes": 10}) == "gpu-compute"
    assert analyze.classify_wait({"head_gpu_util": 95, "head_ib_tx_bytes": 12_000_000, "head_ib_rx_bytes": 12_000_000,
                                  "head_ib_pkts": 18_000, "head_ib_busy_frac": 1.0}) == "fabric-latency"
    assert analyze.classify_wait({"head_gpu_util": 30, "head_ib_tx_bytes": 10_000_000, "head_ib_pkts": 100,
                                  "head_ib_busy_frac": 0.9}) == "fabric-bandwidth"
    assert analyze.classify_wait({"head_gpu_util": 5, "head_ib_tx_bytes": 0}) == "cpu/sched"
    assert analyze.classify_wait({"dur_ms": 70, "client_gap_ms": 500, "head_gpu_util": 30}) == "client"
    assert analyze.classify_wait({"head_gpu_util": 50, "head_ib_tx_bytes": 5, "head_ib_busy_frac": 0.1}) == "mixed"


def _npy_bytes(vals, shape):
    header = "{'descr': '<u2', 'fortran_order': False, 'shape': %s, }" % (tuple(shape),)
    pad = 64 - (10 + len(header) + 1) % 64
    header = header + " " * pad + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin-1") + struct.pack("<" + "H" * len(vals), *vals)


def test_npy_roundtrip_and_expert_stats(tmp_path):
    T, L, K = 4, 3, 2
    vals = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 9, 1, 2, 8, 4, 5, 9, 1, 2, 8, 4, 5]
    p = tmp_path / "x.npy"
    p.write_bytes(_npy_bytes(vals, (T, L, K)))
    shape, descr, got = analyze.load_npy(p)
    assert shape == (T, L, K) and got == vals
    st = analyze.expert_stats(shape, got, prompt_tokens=1)
    assert st["global_distinct_experts"] == 8 and st["distinct_experts_per_token_mean"] == 6.0
    assert st["expert_switch_fraction_mean"] == pytest.approx((0 + 2 / 6 + 0) / 3, abs=1e-3)
    assert len(st["per_layer"]) == 3
    with pytest.raises(ValueError):
        analyze.load_npy(tmp_path / "x.npy") if (p.write_bytes(b"nope") or True) else None


def write_expert_log(host_dir: Path, steps, L=3, K=2, t0=1000.0):
    """Mimic the recorder's file pair. steps: list of (n_rows, sampled, rows[list of [L][K]])."""
    host_dir.mkdir(parents=True, exist_ok=True)
    idx = host_dir / "experts-test-r0-1-20260828T000000Z.idx.jsonl"
    data = host_dir / "experts-test-r0-1-20260828T000000Z.u8"
    blob = bytearray()
    lines = [json.dumps({"type": "meta", "layers": L, "topk": K, "max_tokens": 64})]
    for k, (n, sampled, rows) in enumerate(steps):
        off = len(blob)
        for r in rows:
            for l in range(L):
                blob.extend(bytes(r[l]))
        lines.append(json.dumps({"step": k, "t": t0 + k * 0.07, "n": n, "off": off, "len": len(blob) - off,
                                 "req": ["chatcmpl-1"], "sched": [n], "pos": [k * 2], "draft": [n - 1],
                                 "sampled": [sampled], "rejected": [n - sampled]}))
    idx.write_text("\n".join(lines) + "\n")
    data.write_bytes(bytes(blob))
    return idx, data


def test_expert_log_ingestion_and_stats(tmp_path):
    steps = [
        (2, 1, [[[0, 1], [2, 3], [4, 5]], [[0, 1], [2, 3], [4, 5]]]),      # accepted 1: rows 0 real, row 1 rejected
        (2, 2, [[[9, 1], [2, 8], [4, 5]], [[9, 1], [2, 8], [4, 5]]]),      # both real
    ]
    write_expert_log(tmp_path / "dgx01", steps)
    log = analyze.load_expert_log(tmp_path / "dgx01")
    assert log["meta"]["layers"] == 3 and len(log["steps"]) == 2
    assert analyze.load_expert_log(tmp_path / "dgx01", t0=1000.05, t1=1000.1)["steps"][0]["step"] == 1
    assert analyze.load_expert_log(tmp_path / "nothing") is None
    rows = analyze.expert_rows(log["steps"][0], 3, 2)
    assert rows[0] == [[0, 1], [2, 3], [4, 5]]
    st = analyze.expert_log_stats(log)
    assert st["steps"] == 2 and st["real_tokens"] == 3
    assert st["rows_per_step_mean"] == 2.0 and st["accepted_per_step_mean"] == 1.5
    assert st["distinct_layer_experts_per_step_mean"] == 6.0  # each step's rows share the same 6 pairs
    assert st["global_distinct_layer_experts"] == 8
    assert st["distinct_per_token_mean"] == 6.0
    # token 0 → token 1: 2 of 6 slots changed; token 1 → token 2: 0 → mean 1/6
    assert st["switch_fraction_mean"] == pytest.approx((2 / 6 + 0) / 2, abs=1e-3)
    assert analyze.expert_log_stats(log, req_filter=lambda r: r == "other") is None


def test_phase_summary_between_marks():
    cols = ["ib_tx_bytes", "net_tx_bytes", "gpu_util", "gpu_power_mw", "disk_rd_sectors", "disk_wr_sectors", "pswpin", "pgmajfault"]
    rows = [(float(t), [t * 1_000_000, 0, 10, 10_000, t * 2_000_000, 0, t * 10, t * 5]) for t in range(0, 31)]
    head = _node("dgx01", cols, rows)
    head.marks = [{"t": 5.0, "label": "model_load_start"}, {"t": 25.0, "label": "model_ready"}]
    head.slow = [{"t": 10.0, "mem": {"MemAvailable": 100}, "gpu": {"procs": [{"mem_bytes": 5}]}}]
    ps = analyze.phase_summary({"dgx01": head}, "dgx01", "model_load_start", "model_ready")
    assert ps["dur_s"] == 20.0
    d = ps["dgx01"]
    assert d["disk_rd_GB"] == pytest.approx(20 * 2_000_000 * 512 / 1e9, rel=1e-3)
    assert d["ib_tx_GB"] == pytest.approx(0.02, abs=1e-3)
    assert d["swap_in_pages"] == 200 and d["majfaults"] == 100
    assert len(d["timeline_10s"]) == 2 and d["mem_ramp"] == [(5, 100, 5)]
    assert analyze.phase_summary({"dgx01": head}, "dgx01", "x", "y") is None


def test_analyze_main_end_to_end(tmp_path, capsys):
    """Full CLI over a synthetic trace dir (head + worker + probe + expert log)."""
    root = tmp_path / "traces"
    cols = ["ib_tx_bytes", "ib_rx_bytes", "ib_tx_pkts", "gpu_util", "gpu_mem_util", "gpu_power_mw",
            "disk_rd_sectors", "disk_wr_sectors", "pswpin", "pgmajfault", "net_tx_bytes"]
    for host in ("dgx01", "dgx02"):
        d = root / host
        d.mkdir(parents=True)
        with open(d / "trace-20260828.jsonl", "w") as f:
            f.write(json.dumps({"type": "meta", "t": 0, "host": host, "columns": {"fast": cols}}) + "\n")
            tx = 0
            for i in range(0, 300):
                t = 1000.0 + i * 0.01
                f.write(json.dumps({"type": "fast", "t": t, "host": host, "v": [tx, tx, tx // 700, 90, 0, 24500, 0, 0, 0, 0, 0]}) + "\n")
                tx += 12_800 if t > 1001.0 else 500
            if host == "dgx01":
                for i in range(4):
                    f.write(json.dumps({"type": "vllm", "t": 1000.0 + i, "steps": 100 + i * 7, "gen_tokens": 50 + i * 20,
                                        "accepted_tokens": i * 10, "draft_tokens": i * 20}) + "\n")
                f.write(json.dumps({"type": "clock", "t": 1000.5, "offset_us": 0, "rtt_us": 30}) + "\n")
    with open(root / "dgx01" / "probe-20260828.jsonl", "w") as f:
        f.write(json.dumps({"type": "request", "req": "tt-1", "t": 1000.0, "t_send": 1000.0}) + "\n")
        for i, t in enumerate((1001.0, 1001.07, 1001.14, 1001.21)):
            f.write(json.dumps({"type": "chunk", "req": "tt-1", "t": t, "i": i, "chars": 3}) + "\n")
        f.write(json.dumps({"type": "response", "req": "tt-1", "t": 1001.3, "chunks": 4, "ttfc_s": 1.0, "stream_s": 0.21,
                            "decode_tok_s": 30.0, "usage": {"completion_tokens": 8}}) + "\n")
    write_expert_log(root / "dgx01", [(2, 1, [[[0, 1], [2, 3], [4, 5]]] * 2)] * 4, t0=1001.0)
    out = tmp_path / "out.json"
    md = tmp_path / "out.md"
    rc = analyze.main(["--dir", str(root), "--head", "dgx01", "--worker", "dgx02", "--out", str(out), "--md", str(md)])
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["summary"]["decode_steps"] == 3
    assert res["summary"]["expert_log"]["steps"] == 4
    assert "Expert routing" in md.read_text()
    assert analyze.main(["--dir", str(tmp_path / "empty"), "--head", "dgx01"]) == 2
