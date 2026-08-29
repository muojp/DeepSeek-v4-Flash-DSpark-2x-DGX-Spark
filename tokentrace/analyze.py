"""Merge traces from both nodes and reconstruct per-engine-step activity.

Pure python; the heavy lifting is a handful of interpolations over
cumulative counters. Output: a markdown report on stdout and a JSON file
with the per-step table.
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
import statistics
import struct
from collections import Counter, defaultdict
from pathlib import Path

from .writer import iter_jsonl

# ── loading ──────────────────────────────────────────────────────────────


class NodeTrace:
    def __init__(self, host: str):
        self.host = host
        self.meta: dict | None = None
        self.columns: list[str] = []
        self.fast_t: list[float] = []
        self.fast_v: list[list] = []
        self.slow: list[dict] = []
        self.vllm: list[dict] = []
        self.clock: list[dict] = []
        self.marks: list[dict] = []
        self.errs: list[dict] = []
        self.offset_s = 0.0  # subtract from this node's t to align to the head

    def add(self, rec: dict):
        typ = rec.get("type")
        if typ == "fast":
            self.fast_t.append(rec["t"])
            self.fast_v.append(rec["v"])
        elif typ == "slow":
            self.slow.append(rec)
        elif typ == "vllm":
            self.vllm.append(rec)
        elif typ == "clock":
            self.clock.append(rec)
        elif typ == "mark":
            self.marks.append(rec)
        elif typ == "meta":
            self.meta = rec
            self.columns = rec.get("columns", {}).get("fast", [])
        elif typ == "err":
            self.errs.append(rec)

    def finalize(self):
        order = sorted(range(len(self.fast_t)), key=self.fast_t.__getitem__)
        self.fast_t = [self.fast_t[i] - self.offset_s for i in order]
        self.fast_v = [self.fast_v[i] for i in order]
        for lst in (self.slow, self.vllm, self.marks):
            for r in lst:
                r["t"] = r["t"] - self.offset_s
            lst.sort(key=lambda r: r["t"])

    def col(self, name: str) -> int | None:
        try:
            return self.columns.index(name)
        except ValueError:
            return None

    def value_at(self, name: str, t: float) -> float | None:
        """Linear interpolation of a cumulative counter at wall time t."""
        c = self.col(name)
        if c is None or not self.fast_t:
            return None
        i = bisect.bisect_left(self.fast_t, t)
        if i <= 0:
            return float(self.fast_v[0][c])
        if i >= len(self.fast_t):
            return float(self.fast_v[-1][c])
        t0, t1 = self.fast_t[i - 1], self.fast_t[i]
        v0, v1 = self.fast_v[i - 1][c], self.fast_v[i][c]
        if t1 <= t0:
            return float(v1)
        return v0 + (v1 - v0) * (t - t0) / (t1 - t0)

    def delta(self, name: str, a: float, b: float) -> float | None:
        va, vb = self.value_at(name, a), self.value_at(name, b)
        return None if va is None or vb is None else vb - va

    def mean_gauge(self, name: str, a: float, b: float) -> float | None:
        c = self.col(name)
        if c is None:
            return None
        i = bisect.bisect_left(self.fast_t, a)
        j = bisect.bisect_right(self.fast_t, b)
        vals = [self.fast_v[k][c] for k in range(i, j)]
        if not vals:  # fall back to nearest sample
            k = min(max(i - 1, 0), len(self.fast_v) - 1) if self.fast_v else None
            return None if k is None else float(self.fast_v[k][c])
        return sum(vals) / len(vals)

    def busy_fraction(self, name: str, a: float, b: float, bin_s: float = 0.02, thresh: float = 1.0) -> float | None:
        if self.col(name) is None or b <= a:
            return None
        n = max(1, int((b - a) / bin_s))
        busy = 0
        for k in range(n):
            d = self.delta(name, a + k * bin_s, a + (k + 1) * bin_s)
            if d is not None and d > thresh:
                busy += 1
        return busy / n

    def proc_delta(self, comm_prefix: str, a: float, b: float) -> dict | None:
        """CPU ticks / ctx switches for the first slow sample ≤ a and ≥ b."""
        if not self.slow:
            return None
        ts = [r["t"] for r in self.slow]
        i = max(bisect.bisect_right(ts, a) - 1, 0)
        j = min(bisect.bisect_left(ts, b), len(self.slow) - 1)
        if j <= i:
            return None
        pa, pb = self.slow[i].get("procs", {}), self.slow[j].get("procs", {})
        out = {"utime": 0, "stime": 0, "vcsw": 0, "nvcsw": 0, "span_s": self.slow[j]["t"] - self.slow[i]["t"]}
        found = False
        for pid, s in pb.items():
            if not str(s.get("comm", "")).startswith(comm_prefix) or pid not in pa:
                continue
            found = True
            for k in ("utime", "stime", "vcsw", "nvcsw"):
                if s.get(k) is not None and pa[pid].get(k) is not None:
                    out[k] += s[k] - pa[pid][k]
        return out if found else None


def load_dir(root: str | os.PathLike, hosts: list[str] | None = None, date: str | None = None) -> dict[str, NodeTrace]:
    root = Path(os.path.expanduser(root))
    nodes: dict[str, NodeTrace] = {}
    for hostdir in sorted(root.iterdir() if root.is_dir() else []):
        if not hostdir.is_dir() or (hosts and hostdir.name not in hosts):
            continue
        pattern = f"trace-{date}.jsonl" if date else "trace-*.jsonl"
        files = sorted(glob.glob(str(hostdir / pattern)))
        if not files:
            continue
        nt = NodeTrace(hostdir.name)
        for f in files:
            for rec in iter_jsonl(f):
                nt.add(rec)
        nodes[hostdir.name] = nt
    return nodes


def load_probe(root: str | os.PathLike, host: str, date: str | None = None) -> dict[str, dict]:
    """req_id → {request, chunks:[...], response, routed_experts}"""
    root = Path(os.path.expanduser(root)) / host
    pattern = f"probe-{date}.jsonl" if date else "probe-*.jsonl"
    reqs: dict[str, dict] = defaultdict(lambda: {"chunks": []})
    for f in sorted(glob.glob(str(root / pattern))):
        for rec in iter_jsonl(f):
            rid = rec.get("req")
            if not rid:
                continue
            typ = rec.get("type")
            if typ == "chunk":
                reqs[rid]["chunks"].append(rec)
            elif typ in ("request", "response", "routed_experts"):
                reqs[rid][typ] = rec
    for r in reqs.values():
        r["chunks"].sort(key=lambda c: c["t"])
    return dict(reqs)


def apply_clock_offsets(nodes: dict[str, NodeTrace], head: str):
    """Head's clock records carry offset = peer_clock - head_clock (µs)."""
    h = nodes.get(head)
    if not h or not h.clock:
        for n in nodes.values():
            n.finalize()
        return
    med = statistics.median(c["offset_us"] for c in h.clock) / 1e6
    for name, n in nodes.items():
        if name != head:
            n.offset_s = med
    for n in nodes.values():
        n.finalize()


# ── step reconstruction ──────────────────────────────────────────────────


def step_boundaries_from_vllm(vllm: list[dict]) -> list[tuple[float, float, int]]:
    """(t_a, t_b, n_steps) for consecutive samples where the step counter moved."""
    out = []
    for p, q in zip(vllm, vllm[1:]):
        sp, sq = p.get("steps"), q.get("steps")
        if sp is None or sq is None or sq <= sp:
            continue
        out.append((p["t"], q["t"], int(sq - sp)))
    return out


def step_boundaries_from_chunks(chunks: list[dict], t_send: float | None) -> list[tuple[float, float]]:
    """Each chunk marks the end of an engine step; the first chunk's step
    starts at the request send time (prefill), later ones at the previous
    chunk."""
    if not chunks:
        return []
    out = []
    prev = t_send if t_send is not None else chunks[0]["t"]
    for c in chunks:
        if c["t"] > prev:
            out.append((prev, c["t"]))
        prev = c["t"]
    return out


def classify_wait(row: dict) -> str:
    """Primary wait class for a step. NVML ``gpu_util`` counts NCCL spin
    kernels as busy, so it is *not* used as the compute signal; a step
    whose fabric counters move in ≥ 80 % of its 20 ms bins is fabric-bound
    (latency-bound when the messages are small). Paging / disk activity is
    reported as a flag (``paging_flags``), not a class — it is background
    work whose cost is measured separately (``paging_cost_ms``)."""
    ib = (row.get("head_ib_tx_bytes") or 0) + (row.get("head_ib_rx_bytes") or 0)
    gpu = max(row.get("head_gpu_util") or 0, row.get("worker_gpu_util") or 0)
    busy = row.get("head_ib_busy_frac") or 0
    if row.get("client_gap_ms") and row.get("dur_ms") and row["client_gap_ms"] > 3 * row["dur_ms"]:
        return "client"
    if ib > 0 and busy >= 0.8:
        pk = row.get("head_ib_pkts") or 0
        return "fabric-latency" if pk and ib / pk < 2048 else "fabric-bandwidth"
    if gpu >= 85:
        return "gpu-compute"
    if ib == 0 and gpu < 20:
        return "cpu/sched"
    return "mixed"


def paging_flags(row: dict) -> list[str]:
    flags = []
    for label in ("head", "worker"):
        if (row.get(f"{label}_swap_in") or 0) > 0:
            flags.append(f"{label}-swap-in")
        if (row.get(f"{label}_majfaults") or 0) > 0:
            flags.append(f"{label}-majfault")
        if (row.get(f"{label}_disk_rd_bytes") or 0) > 64 << 10:
            flags.append(f"{label}-disk-read")
    return flags


def build_steps(nodes: dict[str, NodeTrace], head: str, worker: str | None, probe: dict[str, dict]) -> list[dict]:
    h = nodes[head]
    w = nodes.get(worker) if worker else None
    rows: list[dict] = []
    # Prefer probe chunk boundaries (1 chunk = 1 step) when requests exist.
    intervals: list[tuple[float, float, str, int, str]] = []  # (a, b, req, chunk_idx, phase)
    for rid, r in sorted(probe.items(), key=lambda kv: kv[1].get("request", {}).get("t", 0)):
        req = r.get("request")
        if not req or not r["chunks"]:
            continue
        t_send = req.get("t_send", req["t"])
        bounds = step_boundaries_from_chunks(r["chunks"], t_send)
        for k, (a, b) in enumerate(bounds):
            phase = "prefill" if k == 0 else "decode"
            intervals.append((a, b, rid, k, phase))
    if not intervals:
        for a, b, n in step_boundaries_from_vllm(h.vllm):
            intervals.append((a, b, "", n, "unknown"))
    steps_at = [v["t"] for v in h.vllm]
    step_vals = [v.get("steps") for v in h.vllm]
    gen_vals = [v.get("gen_tokens") for v in h.vllm]
    acc_vals = [v.get("accepted_tokens") for v in h.vllm]
    drf_vals = [v.get("draft_tokens") for v in h.vllm]

    def vllm_delta(vals, a, b):
        i = bisect.bisect_right(steps_at, a) - 1
        j = bisect.bisect_right(steps_at, b) - 1
        if i < 0 or j < 0 or vals[i] is None or vals[j] is None:
            return None
        return vals[j] - vals[i]

    prev_end = None
    for idx, (a, b, rid, k, phase) in enumerate(intervals):
        dur = b - a
        row: dict = {"i": idx, "req": rid, "k": k, "phase": phase, "t_start": a, "t_end": b, "dur_ms": round(dur * 1000, 2)}
        row["engine_steps"] = vllm_delta(step_vals, a, b)
        row["gen_tokens"] = vllm_delta(gen_vals, a, b)
        acc, drf = vllm_delta(acc_vals, a, b), vllm_delta(drf_vals, a, b)
        row["accept_ratio"] = round(acc / drf, 3) if acc is not None and drf else None
        for label, node in (("head", h), ("worker", w)):
            if node is None:
                continue
            row[f"{label}_ib_tx_bytes"] = _int(node.delta("ib_tx_bytes", a, b))
            row[f"{label}_ib_rx_bytes"] = _int(node.delta("ib_rx_bytes", a, b))
            row[f"{label}_ib_pkts"] = _int(node.delta("ib_tx_pkts", a, b))
            row[f"{label}_ib_busy_frac"] = _r(node.busy_fraction("ib_tx_bytes", a, b), 3)
            row[f"{label}_net_tx_bytes"] = _int(node.delta("net_tx_bytes", a, b))
            row[f"{label}_gpu_util"] = _r(node.mean_gauge("gpu_util", a, b), 1)
            row[f"{label}_gpu_mem_util"] = _r(node.mean_gauge("gpu_mem_util", a, b), 1)
            row[f"{label}_gpu_power_w"] = _r((node.mean_gauge("gpu_power_mw", a, b) or 0) / 1000, 1)
            rd = node.delta("disk_rd_sectors", a, b)
            wr = node.delta("disk_wr_sectors", a, b)
            row[f"{label}_disk_rd_bytes"] = None if rd is None else int(rd * 512)
            row[f"{label}_disk_wr_bytes"] = None if wr is None else int(wr * 512)
            row[f"{label}_swap_in"] = _int(node.delta("pswpin", a, b))
            row[f"{label}_majfaults"] = _int(node.delta("pgmajfault", a, b))
        row["client_gap_ms"] = None if prev_end is None else round((a - prev_end) * 1000, 2)
        prev_end = b
        row["wait_class"] = classify_wait(row)
        row["paging_flags"] = paging_flags(row)
        rows.append(row)
    return rows


def _int(v):
    return None if v is None else int(round(v))


def _r(v, n):
    return None if v is None else round(v, n)


# ── routed experts (.npy, no numpy) ──────────────────────────────────────


def load_npy(path: str | os.PathLike):
    """Return (shape, dtype_str, flat list of ints) for a C-order integer .npy."""
    raw = Path(path).read_bytes()
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    major = raw[6]
    if major == 1:
        hlen = struct.unpack("<H", raw[8:10])[0]
        hstart = 10
    else:
        hlen = struct.unpack("<I", raw[8:12])[0]
        hstart = 12
    header = raw[hstart:hstart + hlen].decode("latin-1")
    d = eval(header, {"__builtins__": {}}, {})  # noqa: S307 — numpy's own header format (a dict literal)
    shape = tuple(d["shape"])
    descr = d["descr"]
    fortran = d.get("fortran_order", False)
    if fortran:
        raise ValueError("fortran_order not supported")
    fmt = {"|u1": "B", "<u2": "H", "<i4": "i", "<u4": "I", "<i2": "h", "<i8": "q", "<u8": "Q", "|i1": "b"}[descr]
    data = raw[hstart + hlen:]
    n = 1
    for s in shape:
        n *= s
    vals = list(struct.unpack("<" + fmt * n, data[: n * struct.calcsize(fmt)]))
    return shape, descr, vals


def expert_stats(shape, vals, prompt_tokens: int | None = None) -> dict:
    """shape = (tokens, layers, topk)."""
    T, L, K = shape
    per_layer_counts = [Counter() for _ in range(L)]
    distinct_per_token = []
    switch_frac = []
    prev_sets = None
    for t in range(T):
        base = t * L * K
        sets = []
        for l in range(L):
            e = vals[base + l * K: base + (l + 1) * K]
            per_layer_counts[l].update(e)
            sets.append(set(e))
        distinct_per_token.append(sum(len(s) for s in sets))
        if prev_sets is not None:
            same = sum(len(s & p) for s, p in zip(sets, prev_sets))
            switch_frac.append(1 - same / max(1, L * K))
        prev_sets = sets
    layer_summary = []
    for l, c in enumerate(per_layer_counts):
        used = len(c)
        total = sum(c.values())
        top = c.most_common(3)
        # normalised entropy over experts actually routed
        import math
        H = -sum((v / total) * math.log(v / total) for v in c.values()) if total else 0.0
        layer_summary.append({"layer": l, "distinct_experts": used, "entropy_norm": round(H / math.log(max(used, 2)), 3),
                              "top": [[int(e), int(n)] for e, n in top]})
    decode_slice = slice(prompt_tokens or 0, T)
    return {
        "tokens": T, "layers": L, "topk": K,
        "distinct_experts_per_token_mean": round(statistics.mean(distinct_per_token), 2) if distinct_per_token else None,
        "expert_switch_fraction_mean": round(statistics.mean(switch_frac), 3) if switch_frac else None,
        "expert_switch_fraction_decode": round(statistics.mean(switch_frac[decode_slice]), 3) if switch_frac[decode_slice] else None,
        "per_layer": layer_summary,
        "global_distinct_experts": len(set(vals)),
    }


# ── expert log from the V2-runner hotfix (experts-*.idx.jsonl + .u8) ────


def load_expert_log(host_dir: str | os.PathLike, t0: float | None = None, t1: float | None = None) -> dict | None:
    """Steps from every experts-*.idx.jsonl under host_dir whose wall time
    falls in [t0, t1]; the uint8 blobs are read lazily per step."""
    host_dir = Path(host_dir)
    files = sorted(glob.glob(str(host_dir / "experts-*.idx.jsonl")))
    if not files:
        return None
    steps, metas = [], []
    for f in files:
        data_path = f[: -len(".idx.jsonl")] + ".u8"
        try:
            data = open(data_path, "rb").read()
        except OSError:
            continue
        for rec in iter_jsonl(f):
            if rec.get("type") == "meta":
                metas.append(rec)
                continue
            if "step" not in rec:
                continue
            if t0 is not None and rec["t"] < t0:
                continue
            if t1 is not None and rec["t"] > t1:
                continue
            rec["_data"] = data
            steps.append(rec)
    steps.sort(key=lambda r: r["t"])
    return {"meta": metas[-1] if metas else {}, "steps": steps}


def expert_rows(step: dict, L: int, K: int):
    """(n, L, K) nested lists of expert ids for one step."""
    blob = step["_data"][step["off"]: step["off"] + step["len"]]
    n = step["n"]
    out = []
    for i in range(n):
        base = i * L * K
        out.append([list(blob[base + l * K: base + (l + 1) * K]) for l in range(L)])
    return out


def expert_log_stats(log: dict, req_filter=None) -> dict | None:
    """Per-step and per-token expert usage from the runner log.

    A decode step verifies 1 + draft tokens per request; the first
    ``sampled`` rows of a request are real tokens, the rest are rejected
    speculation. Per-step numbers count *all* rows (that is what the
    hardware executed); per-token numbers use real tokens only."""
    meta = log.get("meta") or {}
    L, K = int(meta.get("layers", 43)), int(meta.get("topk", 6))
    steps = [s for s in log["steps"] if req_filter is None or any(req_filter(r) for r in s["req"])]
    if not steps:
        return None
    per_step_distinct, per_step_rows, per_step_accept = [], [], []
    per_layer = [Counter() for _ in range(L)]
    switch, distinct_tok = [], []
    global_seen: set = set()
    prev_sets = None
    real_tokens = 0
    for st in steps:
        rows = expert_rows(st, L, K)
        touched = set()
        for r in rows:
            for l in range(L):
                for e in r[l]:
                    touched.add((l, e))
        per_step_distinct.append(len(touched))
        per_step_rows.append(len(rows))
        global_seen |= touched
        # real tokens: first `sampled` rows of each request span
        off = 0
        acc = 0
        for i, nsched in enumerate(st["sched"]):
            ns = st["sampled"][i] if i < len(st["sampled"]) else 0
            acc += ns
            for j in range(min(ns, nsched)):
                r = rows[off + j]
                sets = [set(r[l]) for l in range(L)]
                for l in range(L):
                    per_layer[l].update(r[l])
                distinct_tok.append(sum(len(x) for x in sets))
                if prev_sets is not None:
                    same = sum(len(a & b) for a, b in zip(sets, prev_sets))
                    switch.append(1 - same / (L * K))
                prev_sets = sets
                real_tokens += 1
            off += nsched
        per_step_accept.append(acc)
    import math
    layer_summary = []
    for l, c in enumerate(per_layer):
        tot = sum(c.values())
        H = -sum((v / tot) * math.log(v / tot) for v in c.values()) if tot else 0.0
        layer_summary.append({"layer": l, "distinct": len(c), "entropy_norm": round(H / math.log(256), 3),
                              "top3": [[int(e), int(n)] for e, n in c.most_common(3)]})
    return {
        "layers": L, "topk": K, "steps": len(steps), "real_tokens": real_tokens,
        "rows_per_step_mean": round(statistics.mean(per_step_rows), 2),
        "accepted_per_step_mean": round(statistics.mean(per_step_accept), 2),
        "accepted_per_step_hist": dict(Counter(per_step_accept)),
        "distinct_layer_experts_per_step_mean": round(statistics.mean(per_step_distinct), 1),
        "distinct_layer_experts_per_step_max": max(per_step_distinct),
        "distinct_layer_experts_total_possible": L * 256,
        "global_distinct_layer_experts": len(global_seen),
        "distinct_per_token_mean": round(statistics.mean(distinct_tok), 2) if distinct_tok else None,
        "switch_fraction_mean": round(statistics.mean(switch), 3) if switch else None,
        "per_layer": layer_summary,
        "least_spread_layers": sorted(layer_summary, key=lambda x: x["entropy_norm"])[:3],
    }


def phase_summary(nodes: dict[str, NodeTrace], head: str, start_label: str, end_label: str) -> dict | None:
    """Totals between two marks on the head (e.g. model_load_start → model_ready)."""
    h = nodes[head]
    t0 = next((m["t"] for m in h.marks if m.get("label") == start_label), None)
    t1 = next((m["t"] for m in h.marks if m.get("label") == end_label and (t0 is None or m["t"] > t0)), None)
    if t0 is None or t1 is None:
        return None
    out = {"start": t0, "end": t1, "dur_s": round(t1 - t0, 1)}
    for label, node in nodes.items():
        rd = node.delta("disk_rd_sectors", t0, t1)
        wr = node.delta("disk_wr_sectors", t0, t1)
        d = {"disk_rd_GB": None if rd is None else round(rd * 512 / 1e9, 2),
             "disk_wr_GB": None if wr is None else round(wr * 512 / 1e9, 2),
             "ib_tx_GB": _r((node.delta("ib_tx_bytes", t0, t1) or 0) / 1e9, 3),
             "net_tx_GB": _r((node.delta("net_tx_bytes", t0, t1) or 0) / 1e9, 3),
             "swap_in_pages": _int(node.delta("pswpin", t0, t1)),
             "majfaults": _int(node.delta("pgmajfault", t0, t1)),
             "gpu_power_w_mean": _r((node.mean_gauge("gpu_power_mw", t0, t1) or 0) / 1000, 1),
             "gpu_util_mean": _r(node.mean_gauge("gpu_util", t0, t1), 1)}
        # 10-second timeline of disk read rate / gpu power / IB tx for the load ramp
        tl = []
        t = t0
        while t < t1:
            u = min(t + 10, t1)
            tl.append({"t": round(t - t0), "disk_rd_MB_s": _r((node.delta("disk_rd_sectors", t, u) or 0) * 512 / (u - t) / 1e6, 1),
                       "ib_tx_MB_s": _r((node.delta("ib_tx_bytes", t, u) or 0) / (u - t) / 1e6, 1),
                       "gpu_w": _r((node.mean_gauge("gpu_power_mw", t, u) or 0) / 1000, 1),
                       "gpu_util": _r(node.mean_gauge("gpu_util", t, u), 0)})
            t = u
        d["timeline_10s"] = tl
        # memory ramp from slow records
        mem = [(round(r["t"] - t0), r["mem"].get("MemAvailable"), (r.get("gpu", {}).get("procs") or [{}])[0].get("mem_bytes"))
               for r in node.slow if t0 <= r["t"] <= t1 and r.get("mem")]
        d["mem_ramp"] = mem[::max(1, len(mem) // 12)]
        out[label] = d
    return out


# ── report ───────────────────────────────────────────────────────────────


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * len(s)))]


def summarize(rows: list[dict], nodes: dict[str, NodeTrace], head: str, worker: str | None, probe: dict) -> dict:
    dec = [r for r in rows if r["phase"] == "decode"]
    pre = [r for r in rows if r["phase"] == "prefill"]
    out: dict = {"steps_total": len(rows), "decode_steps": len(dec), "prefill_steps": len(pre)}

    def agg(rs, key):
        v = [r[key] for r in rs if r.get(key) is not None]
        return {"mean": round(statistics.mean(v), 2), "p50": round(pct(v, .5), 2), "p90": round(pct(v, .9), 2),
                "max": round(max(v), 2), "sum": round(sum(v), 2)} if v else None

    for key in ("dur_ms", "engine_steps", "gen_tokens", "head_ib_tx_bytes", "head_ib_rx_bytes", "head_ib_pkts",
                "head_ib_busy_frac", "head_gpu_util", "head_gpu_mem_util", "head_gpu_power_w", "head_disk_rd_bytes",
                "head_swap_in", "head_majfaults", "worker_ib_tx_bytes", "worker_gpu_util", "worker_gpu_mem_util",
                "worker_disk_rd_bytes", "worker_swap_in", "worker_majfaults", "client_gap_ms", "accept_ratio"):
        a = agg(dec, key)
        if a:
            out.setdefault("decode", {})[key] = a
        a = agg(pre, key)
        if a:
            out.setdefault("prefill", {})[key] = a
    out["wait_classes_decode"] = dict(Counter(r["wait_class"] for r in dec))
    out["paging_flags_decode"] = dict(Counter(f for r in dec for f in r.get("paging_flags", [])))
    # cost of background paging: mean step time with vs without any paging flag
    for label in ("head", "worker"):
        with_p = [r["dur_ms"] for r in dec if any(f.startswith(label) for f in r.get("paging_flags", []))]
        without = [r["dur_ms"] for r in dec if not any(f.startswith(label) for f in r.get("paging_flags", []))]
        if with_p and without:
            out[f"paging_cost_ms_{label}"] = {"steps_with": len(with_p), "dur_with": round(statistics.mean(with_p), 2),
                                              "steps_without": len(without), "dur_without": round(statistics.mean(without), 2),
                                              "delta": round(statistics.mean(with_p) - statistics.mean(without), 2)}
    # step time vs accepted tokens (flat ⇒ fixed per-step cost, MTP tokens are free)
    by_tok: dict[int, list[float]] = defaultdict(list)
    for r in dec:
        if r.get("gen_tokens") is not None:
            by_tok[int(r["gen_tokens"])].append(r["dur_ms"])
    out["dur_by_gen_tokens"] = {k: {"n": len(v), "dur_ms": round(statistics.mean(v), 2)} for k, v in sorted(by_tok.items())}
    if dec:
        tot_dur = sum(r["dur_ms"] for r in dec) / 1000
        tot_tok = sum(r.get("gen_tokens") or 0 for r in dec)
        out["decode_tok_s_from_vllm"] = round(tot_tok / tot_dur, 2) if tot_dur else None
        ib = sum(r.get("head_ib_tx_bytes") or 0 for r in dec)
        out["decode_ib_tx_MB_s"] = round(ib / tot_dur / 1e6, 2) if tot_dur else None
        out["decode_ib_bytes_per_token"] = round(ib / tot_tok) if tot_tok else None
        pk = sum(r.get("head_ib_pkts") or 0 for r in dec)
        out["decode_ib_bytes_per_packet"] = round(ib / pk, 1) if pk else None
    for label in ("head", "worker"):
        pw = [r[f"{label}_gpu_power_w"] for r in dec if r.get(f"{label}_gpu_power_w") is not None]
        pp = [r[f"{label}_gpu_power_w"] for r in pre if r.get(f"{label}_gpu_power_w") is not None]
        if pw or pp:
            out.setdefault("gpu_power_w", {})[label] = {"decode": round(statistics.mean(pw), 1) if pw else None,
                                                        "prefill": round(statistics.mean(pp), 1) if pp else None}
    # per-request summary from probe
    reqs = []
    for rid, r in probe.items():
        resp = r.get("response") or {}
        reqs.append({"req": rid, "chunks": resp.get("chunks"), "ttfc_s": resp.get("ttfc_s"), "stream_s": resp.get("stream_s"),
                     "decode_tok_s": resp.get("decode_tok_s"), "usage": resp.get("usage"),
                     "routed_experts": (r.get("routed_experts") or {}).get("file")})
    out["requests"] = reqs
    h = nodes[head]
    if h.clock:
        out["clock_offset_us_median"] = statistics.median(c["offset_us"] for c in h.clock)
        out["clock_rtt_us_min"] = min(c["rtt_us"] for c in h.clock)
    out["errors"] = {n: Counter(e["source"] for e in nt.errs) for n, nt in nodes.items()}
    return out


def render_markdown(summary: dict, rows: list[dict], experts: dict | None) -> str:
    L = []
    L.append("# tokentrace report\n")
    L.append(f"steps: {summary['steps_total']} (prefill {summary['prefill_steps']}, decode {summary['decode_steps']})")
    if "clock_offset_us_median" in summary:
        L.append(f"clock offset worker−head: {summary['clock_offset_us_median']:.0f} µs (fabric RTT {summary['clock_rtt_us_min']} µs)")
    L.append("")
    for rq in summary.get("requests", []):
        L.append(f"- request `{rq['req']}`: chunks={rq['chunks']} TTFC={rq['ttfc_s']} s stream={rq['stream_s']} s "
                 f"decode={rq['decode_tok_s']} tok/s usage={rq['usage']}")
    L.append("")
    dec = summary.get("decode", {})
    if dec:
        L.append("## Decode steps (1 chunk = 1 engine step)\n")
        L.append("| metric | mean | p50 | p90 | max |")
        L.append("|---|---:|---:|---:|---:|")
        for k, v in dec.items():
            L.append(f"| {k} | {v['mean']} | {v['p50']} | {v['p90']} | {v['max']} |")
        L.append("")
        L.append(f"decode tok/s (vLLM counters over step time): **{summary.get('decode_tok_s_from_vllm')}**; "
                 f"fabric tx {summary.get('decode_ib_tx_MB_s')} MB/s, {summary.get('decode_ib_bytes_per_token')} B/token, "
                 f"{summary.get('decode_ib_bytes_per_packet')} B/packet")
        L.append(f"wait classes: {summary.get('wait_classes_decode')}  (NVML util counts NCCL spin kernels as busy — "
                 f"the fabric-busy fraction and GPU power are the discriminators)")
        L.append(f"paging flags: {summary.get('paging_flags_decode')}")
        for label in ("head", "worker"):
            pc = summary.get(f"paging_cost_ms_{label}")
            if pc:
                L.append(f"- {label} paging cost: {pc['steps_with']} steps with paging avg {pc['dur_with']} ms vs "
                         f"{pc['steps_without']} without avg {pc['dur_without']} ms → Δ {pc['delta']} ms")
        if summary.get("gpu_power_w"):
            L.append(f"- GPU power (W): {summary['gpu_power_w']}")
        if summary.get("dur_by_gen_tokens"):
            L.append("- step time by accepted tokens (vLLM counter, aliased by the 10 Hz poll): " +
                     ", ".join(f"{k}→{v['dur_ms']}ms(n={v['n']})" for k, v in summary["dur_by_gen_tokens"].items()))
        L.append("")
    pre = summary.get("prefill", {})
    if pre:
        L.append("## Prefill (send → first chunk)\n")
        for k in ("dur_ms", "engine_steps", "head_ib_tx_bytes", "head_gpu_util", "head_disk_rd_bytes", "worker_disk_rd_bytes"):
            if k in pre:
                L.append(f"- {k}: {pre[k]}")
        L.append("")
    if experts:
        L.append("## Routed experts\n")
        L.append(f"- tokens×layers×topk = {experts['tokens']}×{experts['layers']}×{experts['topk']}; "
                 f"global distinct experts = {experts['global_distinct_experts']}")
        L.append(f"- distinct (layer,expert) pairs per token: mean {experts['distinct_experts_per_token_mean']} "
                 f"(max possible {experts['layers'] * experts['topk']})")
        L.append(f"- fraction of expert slots that change between consecutive tokens: "
                 f"{experts['expert_switch_fraction_mean']} (decode only: {experts['expert_switch_fraction_decode']})")
        busiest = sorted(experts["per_layer"], key=lambda x: -x["distinct_experts"])[:3]
        L.append(f"- most spread layers: {[(b['layer'], b['distinct_experts'], b['entropy_norm']) for b in busiest]}")
        L.append("")
    el = summary.get("expert_log")
    if el:
        L.append("## Expert routing (runner log, this request)\n")
        L.append(f"- steps {el['steps']}, real tokens {el['real_tokens']}, rows/step {el['rows_per_step_mean']} "
                 f"(1 + drafts), accepted/step {el['accepted_per_step_mean']} hist {el['accepted_per_step_hist']}")
        L.append(f"- distinct (layer, expert) pairs touched per step: mean {el['distinct_layer_experts_per_step_mean']}, "
                 f"max {el['distinct_layer_experts_per_step_max']} of {el['distinct_layer_experts_total_possible']} "
                 f"({100 * el['distinct_layer_experts_per_step_mean'] / el['distinct_layer_experts_total_possible']:.1f} %)")
        L.append(f"- over the whole request: {el['global_distinct_layer_experts']} distinct (layer, expert) pairs "
                 f"({100 * el['global_distinct_layer_experts'] / el['distinct_layer_experts_total_possible']:.1f} % of all experts)")
        L.append(f"- per real token: {el['distinct_per_token_mean']} experts (max {el['layers'] * el['topk']}); "
                 f"slots changing token→token: {el['switch_fraction_mean']}")
        L.append(f"- least spread layers (normalised entropy): {[(x['layer'], x['distinct'], x['entropy_norm']) for x in el['least_spread_layers']]}")
        L.append("")
    lp = summary.get("load_phase")
    if lp:
        L.append("## Model load (model_load_start → model_ready)\n")
        L.append(f"- duration {lp['dur_s']} s")
        for label in ("dgx01", "dgx02"):
            d = lp.get(label)
            if not d:
                continue
            L.append(f"- {label}: NVMe read {d['disk_rd_GB']} GB, written {d['disk_wr_GB']} GB, IB tx {d['ib_tx_GB']} GB, "
                     f"TCP tx {d['net_tx_GB']} GB, swap-in {d['swap_in_pages']} pages, majfaults {d['majfaults']}, "
                     f"GPU {d['gpu_power_w_mean']} W / util {d['gpu_util_mean']} %")
        L.append("")
    errs = summary.get("errors") or {}
    if any(errs.values()):
        L.append(f"errors by source: {errs}")
    return "\n".join(L) + "\n"


def build_parser(p: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    p = p or argparse.ArgumentParser(prog="tokentrace analyze")
    p.add_argument("--dir", default="~/tokentrace")
    p.add_argument("--head", default="dgx01")
    p.add_argument("--worker", default="dgx02")
    p.add_argument("--date", default=None, help="YYYYMMDD (UTC) — default all files")
    p.add_argument("--req", default=None, help="only this request id (substring)")
    p.add_argument("--experts", default=None, help="explicit routed-experts .npy")
    p.add_argument("--out", default=None, help="write per-step JSON here")
    p.add_argument("--md", default=None, help="write markdown report here")
    p.add_argument("--load-start", default="model_load_start")
    p.add_argument("--load-end", default="model_ready")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    nodes = load_dir(a.dir, [a.head, a.worker], a.date)
    if a.head not in nodes:
        print(f"no trace for head {a.head} under {a.dir}")
        return 2
    apply_clock_offsets(nodes, a.head)
    probe = load_probe(a.dir, a.head, a.date)
    if a.req:
        probe = {k: v for k, v in probe.items() if a.req in k}
    rows = build_steps(nodes, a.head, a.worker if a.worker in nodes else None, probe)
    summary = summarize(rows, nodes, a.head, a.worker, probe)
    experts = None
    npy = a.experts
    if not npy:
        for r in probe.values():
            f = (r.get("routed_experts") or {}).get("file")
            if f:
                cand = Path(os.path.expanduser(a.dir)) / a.head / f
                if cand.exists():
                    npy = str(cand)
                    break
    if npy and Path(npy).exists():
        shape, _, vals = load_npy(npy)
        pt = None
        for r in probe.values():
            u = (r.get("routed_experts") or {}).get("usage") or {}
            if u.get("prompt_tokens"):
                pt = u["prompt_tokens"]
        experts = expert_stats(shape, vals, pt)
    # runner-side expert log (hotfix), restricted to the probe request's window
    for rid, r in probe.items():
        resp = r.get("response") or {}
        req = r.get("request") or {}
        if not r["chunks"]:
            continue
        t0, t1 = req.get("t_send", req.get("t")), r["chunks"][-1]["t"] + 0.5
        log = load_expert_log(Path(os.path.expanduser(a.dir)) / a.head, t0 - 0.5, t1)
        if log and log["steps"]:
            st = expert_log_stats(log)
            if st:
                summary["expert_log"] = st
                summary["expert_log_request"] = rid
        break
    lp = phase_summary(nodes, a.head, a.load_start, a.load_end)
    if lp:
        summary["load_phase"] = lp
    md = render_markdown(summary, rows, experts)
    print(md)
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": summary, "steps": rows, "experts": experts}, indent=1), encoding="utf-8")
    if a.md:
        Path(a.md).write_text(md, encoding="utf-8")
    return 0
