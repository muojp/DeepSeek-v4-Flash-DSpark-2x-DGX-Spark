"""Shared pytest fixtures for the tokentrace suite (tests/test_tokentrace_*.py).

Everything runs on any OS: sysfs/procfs are fixture trees, NVML is a fake
ctypes library, vLLM is a local HTTP stub. Nothing here touches the cluster.
"""
from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PIDSTAT = ("306134 (VLLM::Worker_TP0) S 306031 305388 305388 0 -1 4194560 1 2 3 4 123456 6543 0 0 20 0 89 0 4000 "
           "4147000000 983046 18446744073709551615 1 1 0 0 0 0 0 0 0 0 0 0 17 3 0 0 0 0 0 0 0 0 0 0 0 0 0\n")
PIDSTATUS = "Name:\tVLLM::Worker_TP0\nvoluntary_ctxt_switches:\t123\nnonvoluntary_ctxt_switches:\t45\n"
MEMINFO = ("MemTotal:       127600792 kB\nMemFree:         3911832 kB\nMemAvailable:    8853964 kB\n"
           "Cached:          8653888 kB\nSwapCached:      1273248 kB\nAnonPages:       3572592 kB\n"
           "SwapTotal:      16777212 kB\nSwapFree:       13868284 kB\nHugePages_Total:       0\nHugepagesize:       2048 kB\n")
VMSTAT = "nr_free_pages 977958\npswpin 3011\npswpout 7042\npgmajfault 1450\npgfault 987654321\n"
DISKSTATS = (" 259       0 nvme0n1 4157166 206313 847031030 1056712 2545436 2627006 485704138 22450663 0 2184855 23709991 50925 0 7434104536 52318 118935 150296\n"
             " 259       1 nvme0n1p1 100 0 800 1 2 0 16 1 0 1 1 0 0 0 0 0 0\n"
             "   7       0 loop0 1 0 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")
CPUSTAT = "cpu  100 2 300 4000 50 6 7 0 0 0\ncpu0 1 2 3 4 5 6 7 0 0 0\n"
VLLM_METRICS = """# HELP vllm:iteration_tokens_total Histogram of number of tokens per engine_step.
# TYPE vllm:iteration_tokens_total histogram
vllm:iteration_tokens_total_bucket{engine="0",le="1.0",model_name="m"} 100.0
vllm:iteration_tokens_total_count{engine="0",model_name="m"} __STEPS__
vllm:iteration_tokens_total_sum{engine="0",model_name="m"} 987654.0
vllm:generation_tokens_total{engine="0",model_name="m"} 128350.0
vllm:prompt_tokens_total{engine="0",model_name="m"} 5.6e+07
vllm:num_requests_running{engine="0",model_name="m"} __RUNNING__
vllm:num_requests_waiting{engine="0",model_name="m"} 0.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.0012
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} 143845.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 99586.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="0"} 26167.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="2"} 19768.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="1"} 22962.0
vllm:prompt_tokens_by_source_total{engine="0",model_name="m",source="local_compute"} 660051.0
vllm:prompt_tokens_by_source_total{engine="0",model_name="m",source="local_cache_hit"} 5.5909376e+07
vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 5.0
"""


def vllm_metrics_text(steps: float, running: float = 0) -> str:
    return VLLM_METRICS.replace("__STEPS__", str(float(steps))).replace("__RUNNING__", str(float(running)))


def make_fake_roots(base: Path, ib_bytes: int = 1000, net_rx: int = 268013599) -> tuple[Path, Path]:
    """Build a sysfs + procfs tree that looks like a DGX Spark head node."""
    sys_root, proc_root = base / "sys", base / "proc"
    for hca, state, rate in (("rocep1s0f0", "4: ACTIVE", "200 Gb/sec (2X NDR)"),
                             ("rocep1s0f1", "1: DOWN", "40 Gb/sec (4X QDR)"),
                             ("roceP2p1s0f0", "4: ACTIVE", "200 Gb/sec (2X NDR)")):
        p = sys_root / "class/infiniband" / hca / "ports/1"
        (p / "counters").mkdir(parents=True)
        (p / "hw_counters").mkdir(parents=True)
        (p / "state").write_text(state + "\n")
        (p / "rate").write_text(rate + "\n")
        (p / "link_layer").write_text("Ethernet\n")
        (p / "counters/port_rcv_data").write_text(f"{ib_bytes}\n")
        (p / "counters/port_xmit_data").write_text(f"{ib_bytes // 4}\n")
        (p / "counters/port_rcv_packets").write_text("7\n")
        (p / "counters/port_xmit_packets").write_text("8\n")
        (p / "counters/port_xmit_wait").write_text("0\n")
        (p / "hw_counters/rx_write_requests").write_text("99\n")
        (p / "hw_counters/out_of_sequence").write_text("0\n")
    for iface, phys in (("enp1s0f0np0", True), ("lo", False), ("docker0", False), ("wg0", False)):
        d = sys_root / "class/net" / iface
        (d / "statistics").mkdir(parents=True)
        (d / "statistics/rx_bytes").write_text(f"{net_rx}\n")
        (d / "statistics/tx_bytes").write_text("626398482\n")
        (d / "statistics/rx_packets").write_text("1\n")
        (d / "statistics/tx_packets").write_text("2\n")
        (d / "operstate").write_text("up\n")
        if phys:
            (d / "device").mkdir()
    blk = sys_root / "block/nvme0n1"
    blk.mkdir(parents=True)
    (blk / "stat").write_text("4157166 206313 847031030 1056712 2545436 2627006 485704138 22450663 0 2184855 23709991 50925 0 7434104536 52318 118935 150296\n")
    proc_root.mkdir(parents=True)
    (proc_root / "meminfo").write_text(MEMINFO)
    (proc_root / "vmstat").write_text(VMSTAT)
    (proc_root / "diskstats").write_text(DISKSTATS)
    (proc_root / "stat").write_text(CPUSTAT)
    (proc_root / "sys/kernel/random").mkdir(parents=True)
    (proc_root / "sys/kernel/random/boot_id").write_text("a72d0902-4270-46af-b44b-670a9f0a1a23\n")
    for pid, comm, cmd in ((306134, "VLLM::Worker_TP0", "python3"),
                           (305388, "vllm", "/usr/bin/python3 /usr/local/bin/vllm serve x"),
                           (1, "systemd", "/sbin/init"),
                           (4242, "python3", "python3 -m tokentrace sampler")):
        d = proc_root / str(pid)
        d.mkdir(parents=True)
        (d / "comm").write_text(comm + "\n")
        (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode() + b"\0")
        (d / "stat").write_text(PIDSTAT.replace("306134 (VLLM::Worker_TP0)", f"{pid} ({comm})"))
        (d / "status").write_text(PIDSTATUS)
    (proc_root / "notapid").mkdir()
    return sys_root, proc_root


@pytest.fixture
def fake_roots(tmp_path):
    return make_fake_roots(tmp_path)


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _StubHandler(BaseHTTPRequestHandler):
    """Minimal vLLM stand-in: /metrics, /v1/models, /v1/chat/completions (SSE + JSON)."""
    state: dict = {}

    def log_message(self, *_):  # silence
        pass

    def do_GET(self):
        st = self.state
        if self.path.startswith("/metrics"):
            if st.get("metrics_fail"):
                self.send_response(500)
                self.end_headers()
                return
            st["steps"] = st.get("steps", 1000) + st.get("step_inc", 1)
            body = vllm_metrics_text(st["steps"], st.get("running", 0)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/v1/models"):
            self._json({"object": "list", "data": [{"id": "stub"}]})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n) or b"{}")
        st = self.state
        st.setdefault("requests", []).append(req)
        if self.path.startswith("/v1/chat/completions"):
            if st.get("drop"):
                self.connection.close()
                return
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                words = st.get("words", ["Hello", " world", "!", " x", " y"])
                for i, w in enumerate(words):
                    obj = {"choices": [{"delta": {"content": w}, "finish_reason": "length" if i == len(words) - 1 else None}]}
                    self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                    self.wfile.flush()
                    import time
                    time.sleep(st.get("chunk_delay", 0.01))
                usage = {"prompt_tokens": 3, "completion_tokens": len(words), "total_tokens": 3 + len(words)}
                self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": usage}).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self._json({"choices": [{"message": {"content": "hi"}, "routed_experts": st.get("routed_b64")}],
                            "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
        elif self.path.startswith("/tokenize"):
            self._json({"tokens": list(range(len(req.get("prompt", "").split())))})
        else:
            self.send_response(404)
            self.end_headers()

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def vllm_stub():
    """Yields (base_url, state) for a local vLLM stand-in; state is mutable."""
    handler = type("H", (_StubHandler,), {"state": {}})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", handler.state
    finally:
        srv.shutdown()
        srv.server_close()
