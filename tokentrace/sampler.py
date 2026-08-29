"""Node-local sampler daemon (see docs/TOKENTRACE.md §2).

Threads: fast (IB/net/GPU/disk/swap counters, written at ``fast_hz`` while
armed, else 1 Hz), slow (full detail, 1 Hz), vllm (``/metrics`` at
``vllm_hz`` on the head), clock (peer offset, 1 Hz on the head), control
(UDP). Every source is wrapped in an error budget; nothing here can raise
out of its loop.
"""
from __future__ import annotations

import argparse
import os
import platform
import signal
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

from . import SCHEMA_VERSION
from . import control, sources
from .nvml import Nvml
from .writer import TraceWriter

FAST_COLUMNS = [
    "ib_rx_bytes", "ib_tx_bytes", "ib_rx_pkts", "ib_tx_pkts", "ib_xmit_wait",
    "net_rx_bytes", "net_tx_bytes",
    "gpu_util", "gpu_mem_util", "gpu_power_mw", "gpu_sm_mhz",
    "disk_rd_sectors", "disk_wr_sectors", "disk_io_ticks",
    "pswpin", "pswpout", "pgmajfault",
]


class ErrorBudget:
    """After ``limit`` consecutive failures, back off to 1/``divisor`` of the
    calls and report at most once per ``report_s``."""

    def __init__(self, name: str, writer: TraceWriter, limit: int = 10, divisor: int = 10, report_s: float = 60.0):
        self.name, self.writer = name, writer
        self.limit, self.divisor, self.report_s = limit, divisor, report_s
        self.consecutive = 0
        self.total = 0
        self.calls = 0
        self.last_report = 0.0
        self.last_error = ""

    def should_skip(self) -> bool:
        self.calls += 1
        return self.consecutive >= self.limit and (self.calls % self.divisor) != 0

    def ok(self):
        self.consecutive = 0

    def fail(self, err: Exception | str):
        self.consecutive += 1
        self.total += 1
        self.last_error = str(err)[:300]
        now = time.monotonic()
        if self.consecutive == 1 or now - self.last_report >= self.report_s:
            self.last_report = now
            self.writer.write({"type": "err", "source": self.name, "error": self.last_error,
                               "count": self.total, "consecutive": self.consecutive})


class Sampler:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.host = a.host or platform.node().split(".")[0]
        self.sys_root = Path(a.sys_root)
        self.proc_root = Path(a.proc_root)
        self.stop_event = threading.Event()
        self.armed_until = 0.0
        self._arm_lock = threading.Lock()
        outdir = Path(os.path.expanduser(a.dir)) / self.host
        self.writer = TraceWriter(outdir, "trace", self.host, min_free_start=a.min_free_mb << 20,
                                  min_free_run=max(64, a.min_free_mb // 4) << 20)
        self.nvml = Nvml() if not a.no_nvml else Nvml(libname="/nonexistent")
        self.ib_ports = sources.list_ib_ports(self.sys_root)
        self.fast_ib = self._pick_fast_ib(a.hca)
        self.net_ifaces = a.net.split(",") if a.net else sources.list_net_ifaces(self.sys_root)
        self.fast_net = self._pick_fast_net(a.net)
        self.disks = a.disks.split(",") if a.disks else None  # None = auto (whole disks)
        self.fast_disk = None
        self.procs: dict[int, str] = {}
        self._last_proc_scan = 0.0
        self._last_ib_activity_bytes = None
        self.threads: list[threading.Thread] = []
        self.budgets = {n: ErrorBudget(n, self.writer) for n in ("fast", "slow", "vllm", "clock", "nvml", "procs")}
        self.counts = {"fast": 0, "slow": 0, "vllm": 0, "clock": 0, "mark": 0}
        self.vllm_running = 0
        self.control = None

    # ── selection helpers ────────────────────────────────────────────
    def _pick_fast_ib(self, pref: str | None):
        if pref and pref != "auto":
            name, _, port = pref.partition("/")
            return (name, int(port or 1))
        env = os.environ.get("NCCL_IB_HCA", "").split(",")[0].split(":")[0]
        active = [p for p in self.ib_ports if p["state"].upper() == "ACTIVE"]
        for p in active:
            if env and p["name"] == env:
                return (p["name"], p["port"])
        for p in active:
            if not p["name"].startswith("roceP"):  # prefer the on-board port name NCCL_IB_HCA uses
                return (p["name"], p["port"])
        return (active[0]["name"], active[0]["port"]) if active else None

    def _pick_fast_net(self, pref: str | None):
        if pref and pref != "auto":
            return pref.split(",")[0]
        env = os.environ.get("NCCL_SOCKET_IFNAME")
        if env and env in self.net_ifaces:
            return env
        for n in self.net_ifaces:
            if (self.sys_root / "class" / "net" / n / "operstate").exists() and \
               sources.read_text(self.sys_root / "class" / "net" / n / "operstate").strip() == "up" and n.startswith("enp"):
                return n
        return self.net_ifaces[0] if self.net_ifaces else None

    # ── arming ───────────────────────────────────────────────────────
    def arm(self, secs: float, who: str = "local"):
        with self._arm_lock:
            self.armed_until = max(self.armed_until, time.monotonic() + max(0.0, min(secs, 3600.0)))

    @property
    def armed(self) -> bool:
        return time.monotonic() < self.armed_until

    def on_mark(self, label: str, note: str, who: str):
        self.counts["mark"] += 1
        self.writer.write({"type": "mark", "label": label, "note": note, "source": who})
        self.arm(self.a.arm_hold, who)

    def on_stat(self) -> dict:
        return {"host": self.host, "armed": self.armed, "counts": self.counts, "writer": self.writer.stats(),
                "errors": {k: v.total for k, v in self.budgets.items()}, "fast_ib": self.fast_ib,
                "fast_net": self.fast_net, "nvml": self.nvml.available}

    # ── meta ─────────────────────────────────────────────────────────
    def write_meta(self):
        disks = sources.parse_diskstats(sources.read_text(self.proc_root / "diskstats"), self.disks)
        self.fast_disk = self.a.disks.split(",")[0] if self.a.disks else (sorted(disks)[0] if disks else None)
        self.writer.write({
            "type": "meta", "version": SCHEMA_VERSION, "role": self.a.role, "pid": os.getpid(),
            "boot_id": sources.read_text(self.proc_root / "sys/kernel/random/boot_id").strip(),
            "kernel": platform.release(), "python": platform.python_version(),
            "config": {"fast_hz": self.a.fast_hz, "idle_hz": self.a.idle_hz, "slow_hz": self.a.slow_hz, "vllm_hz": self.a.vllm_hz,
                       "vllm_url": self.a.vllm_url if self.a.role == "head" else None,
                       "peer": self.a.peer, "control_port": self.a.control_port, "arm_hold": self.a.arm_hold},
            "hca": self.ib_ports, "fast_ib": self.fast_ib, "net": self.net_ifaces, "fast_net": self.fast_net,
            "disks": sorted(disks), "fast_disk": self.fast_disk,
            "gpu": {"available": self.nvml.available, "name": self.nvml.name, "nvml": self.nvml.version,
                    "supports": self.nvml.supports},
            "columns": {"fast": FAST_COLUMNS},
        })

    # ── loops ────────────────────────────────────────────────────────
    def fast_loop(self):
        period = 1.0 / self.a.fast_hz
        idle_period = 1.0 / max(self.a.slow_hz, 0.1)
        idle_read_period = 1.0 / max(self.a.idle_hz, 0.5)
        last_write = 0.0
        b = self.budgets["fast"]
        tick = 0
        vm = None
        while not self.stop_event.is_set():
            t_start = time.monotonic()
            try:
                ib = sources.read_ib_fast(self.sys_root, *self.fast_ib) if self.fast_ib else (0, 0, 0, 0, 0)
                net = sources.read_net_stats(self.sys_root, self.fast_net) if self.fast_net else {"rx_bytes": 0, "tx_bytes": 0}
                gu = gm = gp = gc = 0
                if self.nvml.available:
                    u = self.nvml.util()
                    if u:
                        gu, gm = u
                    gp = self.nvml.power_mw() or 0
                    gc = self.nvml.sm_mhz() or 0
                dk = sources.read_block_stat(self.sys_root, self.fast_disk) if self.fast_disk else (0, 0, 0)
                # vmstat is ~150 lines; swap/majfault resolution of 100 ms is plenty
                tick += 1
                if tick % self.a.vmstat_every == 1 or vm is None:
                    vm = sources.parse_vmstat(sources.read_text(self.proc_root / "vmstat"), ("pswpin", "pswpout", "pgmajfault"))
                row = [ib[0], ib[1], ib[2], ib[3], ib[4], net["rx_bytes"], net["tx_bytes"], gu, gm, gp, gc,
                       dk[0], dk[1], dk[2], vm.get("pswpin", 0), vm.get("pswpout", 0), vm.get("pgmajfault", 0)]
                # auto-arm on fabric activity
                tot = ib[0] + ib[1]
                if self._last_ib_activity_bytes is not None and tot - self._last_ib_activity_bytes > self.a.arm_ib_bytes:
                    self.arm(self.a.arm_hold, "ib-activity")
                self._last_ib_activity_bytes = tot
                now = time.monotonic()
                if self.armed or now - last_write >= idle_period:
                    self.writer.write({"type": "fast", "v": row})
                    self.counts["fast"] += 1
                    last_write = now
                b.ok()
            except Exception as e:  # noqa: BLE001
                b.fail(e)
            # idle: poll at idle_hz (activity is still noticed within 1/idle_hz s;
            # the probe and the vLLM loop arm explicitly before traffic starts)
            p = period if self.armed else idle_read_period
            self.stop_event.wait(max(0.0, p - (time.monotonic() - t_start)))

    def slow_loop(self):
        period = 1.0 / max(self.a.slow_hz, 0.01)
        b = self.budgets["slow"]
        while not self.stop_event.is_set():
            t_start = time.monotonic()
            try:
                rec = {"type": "slow", "armed": self.armed}
                rec["ib"] = {f"{p['name']}/{p['port']}": sources.read_ib_counters(self.sys_root, p["name"], p["port"])
                             for p in self.ib_ports if p["state"].upper() == "ACTIVE"}
                rec["net"] = {n: sources.read_net_stats(self.sys_root, n) for n in self.net_ifaces}
                rec["mem"] = sources.parse_meminfo(sources.read_text(self.proc_root / "meminfo"))
                rec["vmstat"] = sources.parse_vmstat(sources.read_text(self.proc_root / "vmstat"))
                rec["disk"] = sources.parse_diskstats(sources.read_text(self.proc_root / "diskstats"), self.disks)
                rec["cpu"] = sources.parse_cpu_stat(sources.read_text(self.proc_root / "stat"))
                rec["procs"] = self._proc_samples()
                if self.nvml.available:
                    rec["gpu"] = self._gpu_slow()
                self.writer.write(rec)
                self.counts["slow"] += 1
                b.ok()
            except Exception as e:  # noqa: BLE001
                b.fail(e)
            self.stop_event.wait(max(0.0, period - (time.monotonic() - t_start)))

    def _proc_samples(self) -> dict:
        now = time.monotonic()
        if now - self._last_proc_scan > 30 or not self.procs:
            try:
                self.procs = sources.find_procs(self.proc_root)
                self._last_proc_scan = now
            except Exception as e:  # noqa: BLE001
                self.budgets["procs"].fail(e)
        out = {}
        for pid, comm in list(self.procs.items()):
            s = sources.read_proc_sample(self.proc_root, pid)
            if not s:
                self.procs.pop(pid, None)
                continue
            out[str(pid)] = {"comm": s.get("comm", comm), "utime": s.get("utime"), "stime": s.get("stime"),
                             "rss_kb": (s.get("rss_pages") or 0) * (os.sysconf("SC_PAGE_SIZE") // 1024 if hasattr(os, "sysconf") else 4),
                             "vcsw": s.get("vcsw"), "nvcsw": s.get("nvcsw"), "threads": s.get("num_threads")}
        return out

    def _gpu_slow(self) -> dict:
        b = self.budgets["nvml"]
        try:
            g: dict = {}
            u = self.nvml.util()
            if u:
                g["util"], g["mem_util"] = u
            g["power_mw"] = self.nvml.power_mw()
            g["sm_mhz"] = self.nvml.sm_mhz()
            procs = self.nvml.compute_procs()
            if procs is not None:
                g["procs"] = procs
            now = time.time()
            samples = {}
            for k in self.nvml.supports.get("samples", []):
                s = self.nvml.samples(k)
                if s:
                    samples[k] = [[round(Nvml.sample_ts_to_wall(ts, now), 3), v] for ts, v in s]
            if samples:
                g["samples"] = samples
            b.ok()
            return g
        except Exception as e:  # noqa: BLE001
            b.fail(e)
            return {}

    def vllm_loop(self):
        period = 1.0 / max(self.a.vllm_hz, 0.1)
        url = self.a.vllm_url.rstrip("/") + "/metrics"
        b = self.budgets["vllm"]
        last_steps = None
        last_vllm_write = 0.0
        idle_period = 1.0 / max(self.a.slow_hz, 0.1)
        while not self.stop_event.is_set():
            t_start = time.monotonic()
            if not b.should_skip():
                try:
                    t0 = time.monotonic()
                    with urllib.request.urlopen(url, timeout=2.0) as r:
                        body = r.read().decode("utf-8", errors="replace")
                    lat = (time.monotonic() - t0) * 1000
                    m = sources.parse_vllm_metrics(body)
                    rec = {"type": "vllm", "ok": True, "latency_ms": round(lat, 2)}
                    for k, v in m.items():
                        if isinstance(v, float) and v.is_integer() and k not in ("kv_cache_usage",):
                            v = int(v)
                        rec[k] = v
                    self.vllm_running = int(m.get("running", 0) or 0)
                    if self.vllm_running > 0 or (last_steps is not None and m.get("steps") != last_steps):
                        self.arm(self.a.arm_hold, "vllm-activity")
                    changed = m.get("steps") != last_steps or self.vllm_running > 0
                    last_steps = m.get("steps")
                    now = time.monotonic()
                    # idle server: the counters do not move, so 1 Hz is enough (keeps
                    # the trace ~0.4 MB/h instead of ~25 MB/h on an idle head)
                    if changed or self.armed or now - last_vllm_write >= idle_period:
                        self.writer.write(rec)
                        self.counts["vllm"] += 1
                        last_vllm_write = now
                    b.ok()
                except Exception as e:  # noqa: BLE001
                    b.fail(e)
            self.stop_event.wait(max(0.0, period - (time.monotonic() - t_start)))

    def clock_loop(self):
        b = self.budgets["clock"]
        while not self.stop_event.is_set():
            t_start = time.monotonic()
            if not b.should_skip():
                try:
                    r = control.measure_offset(self.a.peer, self.a.control_port, n=8)
                    if r is None:
                        raise RuntimeError(f"no pong from {self.a.peer}:{self.a.control_port}")
                    self.writer.write({"type": "clock", "peer": self.a.peer, **r})
                    self.counts["clock"] += 1
                    b.ok()
                except Exception as e:  # noqa: BLE001
                    b.fail(e)
            self.stop_event.wait(max(0.0, self.a.clock_period - (time.monotonic() - t_start)))

    # ── lifecycle ────────────────────────────────────────────────────
    def start(self):
        self.write_meta()
        self.control = control.ControlServer(self.a.control_bind, self.a.control_port,
                                             on_arm=self.arm, on_mark=self.on_mark, on_stat=self.on_stat)
        self.control.start()
        self.threads = [threading.Thread(target=self.fast_loop, name="tt-fast", daemon=True),
                        threading.Thread(target=self.slow_loop, name="tt-slow", daemon=True)]
        if self.a.role == "head" and self.a.vllm_url:
            self.threads.append(threading.Thread(target=self.vllm_loop, name="tt-vllm", daemon=True))
        if self.a.peer:
            self.threads.append(threading.Thread(target=self.clock_loop, name="tt-clock", daemon=True))
        for t in self.threads:
            t.start()

    def stop(self, reason: str = "signal"):
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=3.0)
        if self.control:
            self.control.stop()
        try:
            self.writer.write({"type": "mark", "label": "sampler_stop", "note": reason, "source": "self",
                               "stats": self.on_stat()})
        finally:
            self.writer.close()
            self.nvml.shutdown()

    def run(self):
        self.start()
        deadline = time.monotonic() + self.a.duration if self.a.duration else None
        try:
            while not self.stop_event.is_set():
                if deadline and time.monotonic() >= deadline:
                    self.stop("duration")
                    break
                self.stop_event.wait(0.5)
        finally:
            self.stop("exit")


def build_parser(p: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    p = p or argparse.ArgumentParser(prog="tokentrace sampler")
    p.add_argument("--dir", default="~/tokentrace", help="trace root; files land in <dir>/<host>/")
    p.add_argument("--host", default=None, help="override hostname label")
    p.add_argument("--role", choices=("head", "worker"), default="worker")
    p.add_argument("--vllm-url", default="http://127.0.0.1:8888")
    p.add_argument("--peer", default=None, help="worker's fabric IP (head only) for clock offset")
    p.add_argument("--control-bind", default="0.0.0.0")
    p.add_argument("--control-port", type=int, default=control.DEFAULT_PORT)
    p.add_argument("--fast-hz", type=float, default=50.0)
    p.add_argument("--slow-hz", type=float, default=1.0)
    p.add_argument("--vllm-hz", type=float, default=10.0)
    p.add_argument("--clock-period", type=float, default=1.0)
    p.add_argument("--arm-hold", type=float, default=5.0, help="seconds of fast cadence after activity")
    p.add_argument("--idle-hz", type=float, default=10.0, help="fast-loop read rate while not armed")
    p.add_argument("--vmstat-every", type=int, default=5, help="read /proc/vmstat every N fast ticks")
    p.add_argument("--arm-ib-bytes", type=int, default=65536, help="IB bytes per fast tick that count as activity")
    p.add_argument("--hca", default="auto", help="<hca>/<port> for the fast row")
    p.add_argument("--net", default=None, help="comma list of interfaces (first = fast row)")
    p.add_argument("--disks", default=None, help="comma list of block devices (first = fast row)")
    p.add_argument("--sys-root", default="/sys")
    p.add_argument("--proc-root", default="/proc")
    p.add_argument("--min-free-mb", type=int, default=2048)
    p.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = run forever)")
    p.add_argument("--no-nvml", action="store_true")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    s = Sampler(a)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: s.stop_event.set())
    print(f"[tokentrace] sampler {s.host} role={a.role} → {s.writer.path} fast_ib={s.fast_ib} "
          f"fast_net={s.fast_net} nvml={s.nvml.available}", flush=True)
    s.run()
    print(f"[tokentrace] stopped: {s.on_stat()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
