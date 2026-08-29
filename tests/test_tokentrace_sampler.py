"""tokentrace.sampler — the daemon end to end against fixture sysfs/procfs,
a vLLM HTTP stub and the UDP control channel (pytest, no Linux needed)."""
import json
import threading
import time
from pathlib import Path

import pytest

from conftest import free_port
from tokentrace import control
from tokentrace.sampler import FAST_COLUMNS, ErrorBudget, Sampler, build_parser, main
from tokentrace.writer import TraceWriter, iter_jsonl


def _args(fake_roots, tmp_path, vllm_url, port, **over):
    sysr, procr = fake_roots
    argv = ["--dir", str(tmp_path / "trace"), "--host", "testhost", "--role", "head", "--vllm-url", vllm_url,
            "--sys-root", str(sysr), "--proc-root", str(procr), "--no-nvml", "--min-free-mb", "0",
            "--control-bind", "127.0.0.1", "--control-port", str(port),
            "--fast-hz", "50", "--idle-hz", "50", "--slow-hz", "5", "--vllm-hz", "20", "--duration", "1.2"]
    for k, v in over.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return build_parser().parse_args(argv)


def _records(path_dir: Path):
    files = sorted(Path(path_dir).glob("*.jsonl"))
    assert files, "no trace written"
    return [r for f in files for r in iter_jsonl(f)]


def test_sampler_writes_every_record_type(fake_roots, tmp_path, vllm_stub):
    url, state = vllm_stub
    state["running"] = 1  # server busy → sampler arms itself → fast rows at 50 Hz
    port = free_port()
    s = Sampler(_args(fake_roots, tmp_path, url, port))
    assert s.fast_ib == ("rocep1s0f0", 1)  # ACTIVE, non-roceP port preferred
    assert s.fast_net == "enp1s0f0np0"
    s.run()
    recs = _records(tmp_path / "trace" / "testhost")
    by = {}
    for r in recs:
        by.setdefault(r["type"], []).append(r)
    meta = by["meta"][0]
    assert meta["columns"]["fast"] == FAST_COLUMNS and meta["fast_disk"] == "nvme0n1"
    assert meta["hca"][1]["name"] == "rocep1s0f0" and meta["gpu"]["available"] is False
    assert len(by["fast"]) >= 30, len(by["fast"])  # armed → ~50 Hz for 1.2 s
    row = by["fast"][-1]["v"]
    assert len(row) == len(FAST_COLUMNS)
    assert row[FAST_COLUMNS.index("ib_rx_bytes")] == 4000 and row[FAST_COLUMNS.index("net_rx_bytes")] == 268013599
    assert row[FAST_COLUMNS.index("disk_rd_sectors")] == 847031030 and row[FAST_COLUMNS.index("pswpin")] == 3011
    slow = by["slow"][-1]
    assert set(slow["ib"]) == {"rocep1s0f0/1", "roceP2p1s0f0/1"} and slow["ib"]["rocep1s0f0/1"]["rx_bytes"] == 4000
    assert slow["mem"]["MemAvailable"] == 8853964 and slow["disk"]["nvme0n1"]["rd_sectors"] == 847031030
    assert slow["procs"]["306134"]["comm"] == "VLLM::Worker_TP0" and slow["procs"]["306134"]["vcsw"] == 123
    assert "gpu" not in slow  # --no-nvml
    v = by["vllm"][-1]
    assert v["ok"] and v["running"] == 1 and v["steps"] > 1000 and v["accepted_per_pos"] == [26167, 22962, 19768]
    assert len(by["vllm"]) >= 10
    stop = [r for r in by["mark"] if r["label"] == "sampler_stop"]
    assert stop and stop[0]["stats"]["errors"] == {"fast": 0, "slow": 0, "vllm": 0, "clock": 0, "nvml": 0, "procs": 0}
    assert "err" not in by


def test_sampler_idle_throttles_fast_rows(fake_roots, tmp_path, vllm_stub):
    url, state = vllm_stub
    state["running"] = 0
    state["step_inc"] = 0  # counters frozen → never armed
    s = Sampler(_args(fake_roots, tmp_path, url, free_port(), slow_hz=2))
    s.run()
    recs = _records(tmp_path / "trace" / "testhost")
    fast = [r for r in recs if r["type"] == "fast"]
    vllm = [r for r in recs if r["type"] == "vllm"]
    assert 1 <= len(fast) <= 5, len(fast)     # idle → written at slow_hz (2 Hz) for 1.2 s
    assert 1 <= len(vllm) <= 5, len(vllm)     # idle vLLM records throttled the same way


def test_control_arm_mark_and_stat_while_running(fake_roots, tmp_path, vllm_stub):
    url, state = vllm_stub
    state["step_inc"] = 0
    port = free_port()
    s = Sampler(_args(fake_roots, tmp_path, url, port, duration=2.0))
    th = threading.Thread(target=s.run, daemon=True)
    th.start()
    time.sleep(0.3)
    assert control.arm([("127.0.0.1", port)], 5) == {f"127.0.0.1:{port}": True}
    assert s.armed
    assert control.mark([("127.0.0.1", port)], "probe_start", "note") == {f"127.0.0.1:{port}": True}
    st = control.send("127.0.0.1", port, {"op": "stat"})
    assert st["host"] == "testhost" and st["armed"] is True and st["fast_ib"] == ["rocep1s0f0", 1]
    th.join(timeout=5)
    assert not th.is_alive()
    recs = _records(tmp_path / "trace" / "testhost")
    marks = [r for r in recs if r["type"] == "mark" and r["label"] == "probe_start"]
    assert marks and marks[0]["note"] == "note" and marks[0]["source"] == "127.0.0.1"
    fast = [r for r in recs if r["type"] == "fast"]
    assert len(fast) >= 40  # armed for the rest of the run → 50 Hz


def test_vllm_failures_become_err_records_not_crashes(fake_roots, tmp_path, vllm_stub):
    url, state = vllm_stub
    state["metrics_fail"] = True
    s = Sampler(_args(fake_roots, tmp_path, url, free_port()))
    s.run()
    recs = _records(tmp_path / "trace" / "testhost")
    errs = [r for r in recs if r["type"] == "err" and r["source"] == "vllm"]
    assert errs and "500" in errs[0]["error"]
    assert len(errs) <= 2  # reported once, then rate-limited
    assert s.budgets["vllm"].total >= 5 and s.budgets["vllm"].consecutive >= 5
    assert [r for r in recs if r["type"] == "slow"]  # other sources unaffected


def test_error_budget_backoff_and_reporting(tmp_path):
    w = TraceWriter(tmp_path, "t", "h", min_free_start=0, flush_s=0)
    b = ErrorBudget("x", w, limit=3, divisor=10, report_s=1000)
    for i in range(3):
        assert not b.should_skip()
        b.fail(RuntimeError(f"e{i}"))
    skips = [b.should_skip() for _ in range(20)]
    assert skips.count(False) == 2 and skips[0] is True  # 1 in 10 gets through once over the limit
    b.ok()
    assert b.consecutive == 0 and not b.should_skip()
    w.close()
    errs = [r for r in iter_jsonl(w.path) if r["type"] == "err"]
    assert len(errs) == 1 and errs[0]["error"] == "e0" and errs[0]["count"] == 1


def test_pick_fast_ib_honours_nccl_env(fake_roots, tmp_path, vllm_stub, monkeypatch):
    url, _ = vllm_stub
    monkeypatch.setenv("NCCL_IB_HCA", "roceP2p1s0f0:1")
    s = Sampler(_args(fake_roots, tmp_path, url, free_port()))
    assert s.fast_ib == ("roceP2p1s0f0", 1)
    s2 = Sampler(_args(fake_roots, tmp_path, url, free_port(), hca="rocep1s0f1/1"))
    assert s2.fast_ib == ("rocep1s0f1", 1)
    for x in (s, s2):
        x.writer.close()


def test_refuses_to_start_without_disk_space(fake_roots, tmp_path, vllm_stub):
    url, _ = vllm_stub
    with pytest.raises(RuntimeError):
        Sampler(_args(fake_roots, tmp_path, url, free_port(), min_free_mb=1 << 30))


def test_main_entry_returns_zero(fake_roots, tmp_path, vllm_stub, capsys):
    url, _ = vllm_stub
    sysr, procr = fake_roots
    rc = main(["--dir", str(tmp_path / "t"), "--host", "h", "--role", "worker", "--sys-root", str(sysr), "--proc-root", str(procr),
               "--no-nvml", "--min-free-mb", "0", "--control-bind", "127.0.0.1", "--control-port", str(free_port()),
               "--duration", "0.3", "--slow-hz", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sampler h role=worker" in out and "stopped:" in out
    recs = _records(tmp_path / "t" / "h")
    assert not [r for r in recs if r["type"] == "vllm"]  # worker role: no vLLM loop
    assert json.dumps([r for r in recs if r["type"] == "meta"][0]["config"])  # serialisable
