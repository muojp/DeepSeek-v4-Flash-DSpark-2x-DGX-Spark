"""Read-only data sources: pure parsers + thin sysfs/procfs readers.

Every parser takes text and returns plain dicts of ints so it can be unit
tested on any OS. Readers take an explicit root (``/sys``, ``/proc``) so
tests can point them at fixture trees.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# ── generic ──────────────────────────────────────────────────────────────


def read_text(path: str | os.PathLike, default: str = "") -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return default


def read_int(path: str | os.PathLike, default: int | None = None) -> int | None:
    s = read_text(path, "").strip()
    if not s:
        return default
    try:
        return int(s.split()[0])
    except ValueError:
        return default


# ── /proc/meminfo, /proc/vmstat ──────────────────────────────────────────

MEMINFO_KEYS = (
    "MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "SwapCached",
    "AnonPages", "Mapped", "Shmem", "Dirty", "Writeback", "SwapTotal", "SwapFree",
    "Slab", "Unevictable", "Mlocked", "Committed_AS", "HugePages_Total",
)
VMSTAT_KEYS = ("pswpin", "pswpout", "pgmajfault", "pgfault", "pgpgin", "pgpgout")


def parse_meminfo(text: str, keys=MEMINFO_KEYS) -> dict[str, int]:
    """Values in kB (as the kernel prints them)."""
    out: dict[str, int] = {}
    want = set(keys)
    for line in text.splitlines():
        k, sep, rest = line.partition(":")
        if not sep or k not in want:
            continue
        parts = rest.split()
        if parts:
            try:
                out[k] = int(parts[0])
            except ValueError:
                pass
    return out


def parse_vmstat(text: str, keys=VMSTAT_KEYS) -> dict[str, int]:
    out: dict[str, int] = {}
    want = set(keys)
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in want:
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return out


# ── /proc/diskstats ──────────────────────────────────────────────────────

DISK_FIELDS = ("rd_ios", "rd_merges", "rd_sectors", "rd_ticks",
               "wr_ios", "wr_merges", "wr_sectors", "wr_ticks",
               "in_flight", "io_ticks", "time_in_queue")


def parse_diskstats(text: str, devices: list[str] | None = None) -> dict[str, dict[str, int]]:
    """Per-device counters. Sectors are always 512 B. ``devices=None`` keeps
    every whole disk that looks like nvme*n* / sd? / mmcblk* (no partitions)."""
    out: dict[str, dict[str, int]] = {}
    whole = re.compile(r"^(nvme\d+n\d+|sd[a-z]+|vd[a-z]+|mmcblk\d+)$")
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 14:
            continue
        name = parts[2]
        if devices is not None:
            if name not in devices:
                continue
        elif not whole.match(name):
            continue
        try:
            vals = [int(x) for x in parts[3:14]]
        except ValueError:
            continue
        out[name] = dict(zip(DISK_FIELDS, vals))
    return out


def read_block_stat(sys_root: str | os.PathLike, dev: str) -> tuple:
    """(rd_sectors, wr_sectors, io_ticks) from /sys/block/<dev>/stat — one
    short line instead of the whole /proc/diskstats (hot path)."""
    parts = read_text(Path(sys_root) / "block" / dev / "stat", "").split()
    if len(parts) < 10:
        return (0, 0, 0)
    try:
        return (int(parts[2]), int(parts[6]), int(parts[9]))
    except ValueError:
        return (0, 0, 0)


# ── /proc/stat, /proc/<pid>/stat, /proc/<pid>/status ─────────────────────

CPU_FIELDS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")


def parse_cpu_stat(text: str) -> dict[str, int]:
    for line in text.splitlines():
        if line.startswith("cpu "):
            parts = line.split()[1:]
            vals = []
            for p in parts[: len(CPU_FIELDS)]:
                try:
                    vals.append(int(p))
                except ValueError:
                    vals.append(0)
            return dict(zip(CPU_FIELDS, vals))
    return {}


def parse_pid_stat(text: str) -> dict:
    """/proc/<pid>/stat → comm, state, utime, stime (clock ticks), rss (pages),
    num_threads. Handles parentheses in comm."""
    lp = text.find("(")
    rp = text.rfind(")")
    if lp < 0 or rp < 0:
        return {}
    comm = text[lp + 1:rp]
    rest = text[rp + 2:].split()
    # rest[0] = state (field 3); utime = field 14 → rest[11]; stime → rest[12]
    try:
        return {
            "comm": comm,
            "state": rest[0],
            "utime": int(rest[11]),
            "stime": int(rest[12]),
            "num_threads": int(rest[17]),
            "rss_pages": int(rest[21]),
        }
    except (IndexError, ValueError):
        return {"comm": comm}


def parse_pid_status_ctxt(text: str) -> dict[str, int]:
    out = {}
    for line in text.splitlines():
        if line.startswith("voluntary_ctxt_switches:"):
            out["vcsw"] = int(line.split()[1])
        elif line.startswith("nonvoluntary_ctxt_switches:"):
            out["nvcsw"] = int(line.split()[1])
    return out


def find_procs(proc_root: str | os.PathLike, comm_prefixes=("VLLM::",), name_substrings=("vllm",)) -> dict[int, str]:
    """pid → comm for processes whose comm starts with a prefix or whose
    cmdline contains a substring (``vllm serve``)."""
    found: dict[int, str] = {}
    root = Path(proc_root)
    try:
        entries = os.listdir(root)
    except OSError:
        return found
    for e in entries:
        if not e.isdigit():
            continue
        pid = int(e)
        comm = read_text(root / e / "comm", "").strip()
        if any(comm.startswith(p) for p in comm_prefixes):
            found[pid] = comm
            continue
        if name_substrings:
            cmd = read_text(root / e / "cmdline", "").replace("\0", " ")
            if any(s in cmd for s in name_substrings) and "tokentrace" not in cmd:
                found[pid] = comm or "?"
    return found


def read_proc_sample(proc_root: str | os.PathLike, pid: int) -> dict:
    root = Path(proc_root) / str(pid)
    st = parse_pid_stat(read_text(root / "stat", ""))
    if not st:
        return {}
    st.update(parse_pid_status_ctxt(read_text(root / "status", "")))
    return st


# ── InfiniBand / RoCE ────────────────────────────────────────────────────

IB_DATA_UNIT = 4  # port_{rcv,xmit}_data are in 4-byte (32-bit word) units
IB_COUNTERS = {  # sysfs file → record key
    "port_rcv_data": "rx_bytes",   # ×4 applied
    "port_xmit_data": "tx_bytes",  # ×4 applied
    "port_rcv_packets": "rx_pkts",
    "port_xmit_packets": "tx_pkts",
    "port_xmit_wait": "xmit_wait",
    "port_rcv_errors": "rx_errors",
    "port_xmit_discards": "tx_discards",
}
IB_HW_COUNTERS = {
    "rx_write_requests": "rx_write_req",
    "rx_read_requests": "rx_read_req",
    "out_of_sequence": "out_of_sequence",
    "packet_seq_err": "packet_seq_err",
    "local_ack_timeout_err": "local_ack_timeout_err",
    "np_cnp_sent": "np_cnp_sent",
    "rp_cnp_handled": "rp_cnp_handled",
    "np_ecn_marked_roce_packets": "ecn_marked",
}


def list_ib_ports(sys_root: str | os.PathLike) -> list[dict]:
    base = Path(sys_root) / "class" / "infiniband"
    ports: list[dict] = []
    try:
        hcas = sorted(os.listdir(base))
    except OSError:
        return ports
    for hca in hcas:
        pdir = base / hca / "ports"
        try:
            pnums = sorted(os.listdir(pdir), key=lambda s: int(s) if s.isdigit() else 0)
        except OSError:
            continue
        for p in pnums:
            d = pdir / p
            state = read_text(d / "state", "").strip()
            ports.append({
                "name": hca,
                "port": int(p) if p.isdigit() else p,
                "rate": read_text(d / "rate", "").strip(),
                "state": state.split(":", 1)[-1].strip() if state else "",
                "link_layer": read_text(d / "link_layer", "").strip(),
            })
    return ports


def parse_ib_rate_mbps(rate: str) -> int | None:
    m = re.match(r"\s*([\d.]+)\s*(Gb|Mb)/sec", rate or "")
    if not m:
        return None
    v = float(m.group(1))
    return int(v * 1000) if m.group(2) == "Gb" else int(v)


def read_ib_counters(sys_root: str | os.PathLike, hca: str, port: int | str, hw: bool = True) -> dict[str, int]:
    d = Path(sys_root) / "class" / "infiniband" / hca / "ports" / str(port)
    out: dict[str, int] = {}
    for f, key in IB_COUNTERS.items():
        v = read_int(d / "counters" / f)
        if v is None:
            continue
        out[key] = v * IB_DATA_UNIT if f.endswith("_data") else v
    if hw:
        for f, key in IB_HW_COUNTERS.items():
            v = read_int(d / "hw_counters" / f)
            if v is not None:
                out[key] = v
    return out


def read_ib_fast(sys_root: str | os.PathLike, hca: str, port: int | str) -> tuple:
    """(rx_bytes, tx_bytes, rx_pkts, tx_pkts, xmit_wait) — the hot path."""
    d = Path(sys_root) / "class" / "infiniband" / hca / "ports" / str(port) / "counters"
    rx = read_int(d / "port_rcv_data", 0) * IB_DATA_UNIT
    tx = read_int(d / "port_xmit_data", 0) * IB_DATA_UNIT
    return (rx, tx, read_int(d / "port_rcv_packets", 0), read_int(d / "port_xmit_packets", 0),
            read_int(d / "port_xmit_wait", 0))


# ── /sys/class/net ───────────────────────────────────────────────────────


def list_net_ifaces(sys_root: str | os.PathLike, physical_only: bool = True) -> list[str]:
    base = Path(sys_root) / "class" / "net"
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return []
    out = []
    for n in names:
        if n == "lo" or n.startswith(("docker", "veth", "br-", "virbr", "wg", "tailscale")):
            continue
        if physical_only and not (base / n / "device").exists():
            continue
        out.append(n)
    return out


def read_net_stats(sys_root: str | os.PathLike, iface: str) -> dict[str, int]:
    d = Path(sys_root) / "class" / "net" / iface / "statistics"
    return {
        "rx_bytes": read_int(d / "rx_bytes", 0),
        "tx_bytes": read_int(d / "tx_bytes", 0),
        "rx_packets": read_int(d / "rx_packets", 0),
        "tx_packets": read_int(d / "tx_packets", 0),
    }


# ── vLLM /metrics ────────────────────────────────────────────────────────

_VLLM_SINGLE = {
    "steps": "vllm:iteration_tokens_total_count",
    "step_tokens": "vllm:iteration_tokens_total_sum",
    "gen_tokens": "vllm:generation_tokens_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "prompt_cached": "vllm:prompt_tokens_cached_total",
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "running": "vllm:num_requests_running",
    "waiting": "vllm:num_requests_waiting",
    "kv_cache_usage": "vllm:kv_cache_usage_perc",
    "preemptions": "vllm:num_preemptions_total",
    "requests_success": "vllm:request_success_total",
    "prefix_hits": "vllm:prefix_cache_hits_total",
    "prefix_queries": "vllm:prefix_cache_queries_total",
}
_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|Inf)\s*$")
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_vllm_metrics(text: str) -> dict:
    """Aggregate over label sets (sum) for the counters/gauges we track;
    ``accepted_per_pos`` is the per-position list, ``prompt_by_source`` a
    dict. Values are floats (Prometheus exposition)."""
    wanted = {v: k for k, v in _VLLM_SINGLE.items()}
    out: dict = {}
    per_pos: dict[int, float] = {}
    by_source: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        try:
            v = float(val)
        except ValueError:
            continue
        if name in wanted:
            k = wanted[name]
            out[k] = out.get(k, 0.0) + v
        elif name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            lm = dict(_LABEL.findall(labels))
            try:
                per_pos[int(lm.get("position", "-1"))] = per_pos.get(int(lm.get("position", "-1")), 0.0) + v
            except ValueError:
                pass
        elif name == "vllm:prompt_tokens_by_source_total":
            lm = dict(_LABEL.findall(labels))
            src = lm.get("source", "?")
            by_source[src] = by_source.get(src, 0.0) + v
    if per_pos:
        n = max(per_pos) + 1
        out["accepted_per_pos"] = [per_pos.get(i, 0.0) for i in range(n)]
    if by_source:
        out["prompt_by_source"] = by_source
    return out
