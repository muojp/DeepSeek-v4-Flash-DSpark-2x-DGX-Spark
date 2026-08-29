"""Render a tokentrace run as a video (PIL frames → ffmpeg).

Inputs (all produced by the sampler / probe / analyze / expert hotfix):
  --steps   analyze --out JSON (per-chunk rows with fabric/GPU/paging)
  --chunks  <req>-chunks.json  (per-chunk text + token ids, from /tokenize)
  --experts host dir with experts-*.idx.jsonl + .u8 (runner log)

What the picture encodes (see docs/TOKENTRACE.md §7):
  * big matrix = 43 layers × 256 routed experts, one cell per expert. Both
    nodes compute every routed expert (each holds half of its intermediate
    width, TP=2), so the map is identical on dgx01 and dgx02 — no per-node
    split inside a cell; node state is carried by the edge strips only;
  * brightness = recency-weighted use (saturates after ~3 uses, decays with
    a τ of a few seconds of generation time, never-used stays dark);
  * cells hit by a step flash white when they served an accepted token and
    amber when they only served a rejected draft (wasted work); the flash
    decays over ~4 video frames regardless of playback speed;
  * a node half is tinted red while that node is paging (swap-in / major
    fault) in the current step;
  * step class: FAST (≥5 accepted and step ≤ 1.1× baseline), STALL (step
    > 1.15× baseline — a time excursion, ≈ the slowest 10 %), else NORMAL — colours green /
    orange-red / grey-blue throughout; paging and rejected drafts are shown
    as tints/flashes, not as a class, because they cost ~1 ms per step;
  * machine rows: dgx01 / dgx02 stacked, one value per column for the current
    engine step from the 50 Hz samplers (RoCE tx/rx MB, tx Gb/s, packets, GPU
    W, util, swap-in, majfaults, NVMe read) with a 30 s per-step mini graph
    under each value and the 30 s cumulative count next to swap-in /
    majfault; System W (sparkDash estimate GPU+CPU+peripherals) and GPU
    temperature come from Prometheus 5 s samples (step-held) when --grafana
    is given;
  * bottom: the generated text streams in, coloured by the step class of the
    step that produced it; for the newest accepted token, per-layer bars show
    how many of its 6 experts the previous token also used (routing
    continuity: ~36 % vs 2 % random; shallow layers re-route every token).
    Weights are read per step, so the cost-relevant number is the distinct
    (layer, expert) pairs of the step vs its rows × 258 slots.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .analyze import load_expert_log

W, H = 1920, 1080
BG = (11, 15, 20)
FG = (222, 226, 232)
DIM = (120, 128, 140)
C_FAST = (70, 200, 120)
C_NORMAL = (110, 140, 190)
C_STALL = (240, 140, 60)
C_PAGING = (230, 70, 70)
C_ACCEPT_FLASH = (255, 255, 255)
C_REJECT_FLASH = (255, 190, 90)
NEVER = np.array([22, 26, 32], dtype=np.float32)
LO = np.array([8, 36, 50], dtype=np.float32)
HI = np.array([90, 230, 255], dtype=np.float32)

L_LAYERS, N_EXPERTS, TOPK = 43, 256, 6


def font(size: int, mono: bool = True):
    for p in (["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/SFNSMono.ttf"] if mono else
              ["/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial Unicode.ttf"]):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ── data model ───────────────────────────────────────────────────────────


class Step:
    __slots__ = ("k", "t", "dur", "accepted", "rows", "experts", "text", "tokens", "cls", "ib_mb", "pkts",
                 "pw_head", "pw_worker", "paging_head", "paging_worker", "phase", "n_distinct", "raw")

    def __init__(self):
        self.experts = None


def build(steps_json: str, chunks_json: str, experts_dir: str, req_substr: str) -> list[Step]:
    an = json.load(open(steps_json))
    rows = [r for r in an["steps"] if req_substr in (r.get("req") or "")]
    rows.sort(key=lambda r: r["k"])
    chunks = json.load(open(chunks_json))
    assert len(chunks) == len(rows), (len(chunks), len(rows))
    t0 = rows[0]["t_start"]
    t_end = rows[-1]["t_end"]
    log = load_expert_log(experts_dir, t0 - 1.0, t_end + 1.0)
    assert log and log["steps"], "no expert log in window"
    L, K = int(log["meta"].get("layers", L_LAYERS)), int(log["meta"].get("topk", TOPK))
    # keep the single request that dominates the window
    from collections import Counter
    rid = Counter(r for s in log["steps"] for r in s["req"]).most_common(1)[0][0]
    lsteps = [s for s in log["steps"] if s["req"] == [rid]]
    # align: chunk k ↔ log step k (one chunk per engine step); tolerate a
    # missing tail by nearest-time matching
    out: list[Step] = []
    for k, (r, c) in enumerate(zip(rows, chunks)):
        st = Step()
        st.k, st.t, st.dur = k, r["t_end"], r["dur_ms"]
        st.phase = r["phase"]
        st.raw = r
        ls = lsteps[k] if k < len(lsteps) else min(lsteps, key=lambda s: abs(s["t"] - r["t_end"]))
        blob = ls["_data"][ls["off"]: ls["off"] + ls["len"]]
        arr = np.frombuffer(blob, dtype=np.uint8).reshape(ls["n"], L, K)
        st.rows = ls["n"]
        st.accepted = int(ls["sampled"][0]) if ls["sampled"] else 1
        st.experts = arr
        st.text, st.tokens = c["text"], c["tokens"]
        st.ib_mb = ((r.get("head_ib_tx_bytes") or 0) + (r.get("head_ib_rx_bytes") or 0)) / 1e6
        st.pkts = r.get("head_ib_pkts") or 0
        st.pw_head, st.pw_worker = r.get("head_gpu_power_w") or 0, r.get("worker_gpu_power_w") or 0
        fl = r.get("paging_flags") or []
        st.paging_head = any(f.startswith("head") for f in fl)
        st.paging_worker = any(f.startswith("worker") for f in fl)
        touched = set()
        for i in range(arr.shape[0]):
            for l in range(L):
                for e in arr[i, l]:
                    touched.add((l, int(e)))
        st.n_distinct = len(touched)
        out.append(st)
    dec = [s.dur for s in out if s.phase == "decode"]
    base = float(np.median(dec)) if dec else 67.0
    for s in out:
        if s.phase == "prefill":
            s.cls = "PREFILL"
        elif s.dur > 1.15 * base:
            s.cls = "STALL"  # time excursion (≈ slowest 10 %): fabric wait / paging / scheduler
        elif s.accepted >= 5 and s.dur <= 1.1 * base:
            s.cls = "FAST"
        else:
            s.cls = "NORMAL"
    return out


def cls_color(c: str):
    return {"FAST": C_FAST, "STALL": C_STALL, "NORMAL": C_NORMAL, "PREFILL": (180, 160, 220)}[c]


# ── machine stats from Prometheus (Grafana fleet overview, 5 s) ──────────


class MachineStats:
    """Series pulled from Prometheus for the run window:
    {"series": {name: {host: [[t, v], ...]}}}. Linear interpolation in time."""
    PANELS = (("gpu_temp_c", "GPU temp", "°C", "{:3.0f}"), ("gpu_power_w", "GPU power", "W", "{:5.1f}"),
              ("rdma_rx_gbps", "RDMA rx", "Gb/s", "{:5.2f}"), ("rdma_tx_gbps", "RDMA tx", "Gb/s", "{:5.2f}"))

    def __init__(self, path: str | None):
        self.series: dict = {}
        if path:
            self.series = json.load(open(path)).get("series", {})

    @property
    def available(self) -> bool:
        return bool(self.series)

    def hosts(self, name: str) -> list[str]:
        return sorted(self.series.get(name, {}))

    def value_at(self, name: str, host: str, t: float) -> float | None:
        pts = self.series.get(name, {}).get(host)
        if not pts:
            return None
        # step-hold: the last sample at or before t (no interpolation, no smoothing)
        if t < pts[0][0]:
            return pts[0][1]
        v = pts[0][1]
        for ts, val in pts:
            if ts <= t:
                v = val
            else:
                break
        return v

    def changed_at(self, name: str, host: str, t: float) -> float | None:
        """Wall time of the most recent sample at or before t (for change flashes)."""
        pts = self.series.get(name, {}).get(host)
        if not pts:
            return None
        last = None
        for ts, _ in pts:
            if ts <= t:
                last = ts
            else:
                break
        return last

    def bounds(self, name: str, t0: float, t1: float) -> tuple[float, float]:
        vals = [v for h in self.series.get(name, {}).values() for t, v in h if t0 - 10 <= t <= t1 + 10]
        if not vals:
            return 0.0, 1.0
        lo, hi = min(vals), max(vals)
        return (lo, hi) if hi > lo else (lo - 1, hi + 1)


# ── renderer ─────────────────────────────────────────────────────────────


class Renderer:
    def __init__(self, steps: list[Step], slowmo: float, fps: int, tau: float, flash_frames: float = 4.0,
                 stats: "MachineStats | None" = None):
        self.steps = steps
        self.stats = stats or MachineStats(None)
        self.slowmo, self.fps, self.tau = slowmo, fps, tau
        # a flash decays to ~10 % after `flash_frames` video frames, whatever the playback speed
        self.flash_tau = (flash_frames / (fps * slowmo)) / 2.3
        self.t0 = steps[0].t - steps[0].dur / 1000.0  # request send time
        self.t_end = steps[-1].t + 1.0
        self.heat = np.zeros((L_LAYERS, N_EXPERTS), dtype=np.float32)
        self.ever = np.zeros((L_LAYERS, N_EXPERTS), dtype=bool)
        self.applied = 0  # steps folded into heat
        self.last_heat_t = self.t0
        self.f_title = font(30, mono=False)
        self.f_big = font(40)
        self.f = font(20)
        self.f_mid = font(28)
        self.f_small = font(16)
        self.f_tok = font(26)
        # layout
        self.map_x, self.map_y = 96, 118
        self.cw, self.ch = 7, 7  # cell: 6 px + 1 gap ; row 6 px + 1 gap (one cell per expert — both nodes compute every expert)
        self.map_w, self.map_h = N_EXPERTS * self.cw, L_LAYERS * self.ch
        self.prev_tok = None          # [L, K] experts of the previous accepted token
        self.last_overlap = None      # per layer: how many of the newest token's K experts the previous token also used
        self.last_tok_text = ""
        self.stream_lines: list[list[tuple[str, tuple]]] = [[]]
        self.stream_applied = 0
        self.total_tokens = sum(s.accepted for s in steps)

    # heat update (called once per new step, in order)
    def _apply_step(self, s: Step):
        dt = max(0.0, s.t - self.last_heat_t)
        self.heat *= math.exp(-dt / self.tau)
        self.last_heat_t = s.t
        acc = s.experts[: s.accepted]
        for i in range(acc.shape[0]):
            for l in range(L_LAYERS):
                for e in acc[i, l]:
                    self.heat[l, e] = min(1.0, self.heat[l, e] + 0.25)
                    self.ever[l, e] = True
            if self.prev_tok is not None:
                self.last_overlap = [len(set(acc[i, l].tolist()) & set(self.prev_tok[l].tolist())) for l in range(L_LAYERS)]
            self.prev_tok = acc[i]
        if acc.shape[0]:
            self.last_tok_text = s.text

    def _current_sets(self, s: Step):
        acc = np.zeros((L_LAYERS, N_EXPERTS), dtype=bool)
        rej = np.zeros((L_LAYERS, N_EXPERTS), dtype=bool)
        for i in range(s.experts.shape[0]):
            tgt = acc if i < s.accepted else rej
            for l in range(L_LAYERS):
                tgt[l, s.experts[i, l]] = True
        rej &= ~acc
        return acc, rej

    def _heat_image(self, s: Step | None, t_real: float) -> Image.Image:
        # decay to the current instant (visual only)
        dt = max(0.0, t_real - self.last_heat_t)
        heat = self.heat * math.exp(-dt / self.tau)
        v = heat[..., None]
        rgb = np.where(self.ever[..., None], LO + (HI - LO) * v, NEVER)
        img = np.zeros((self.map_h, self.map_w, 3), dtype=np.float32)
        img[:] = BG
        cell = np.repeat(np.repeat(rgb, self.ch, axis=0), self.cw, axis=1)
        # flashes of the last two steps, each fading with flash_tau so a step
        # stays visible for ~4 frames at any playback speed (oldest drawn first)
        for st in self.steps[max(0, self.applied - 2): self.applied]:
            fade = math.exp(-max(0.0, t_real - st.t) / self.flash_tau)
            if fade < 0.05:
                continue
            acc, rej = self._current_sets(st)
            flash = np.zeros_like(rgb)
            flash[acc] = C_ACCEPT_FLASH
            flash[rej] = C_REJECT_FLASH
            fm = (acc | rej)[..., None]
            cellf = np.repeat(np.repeat(np.where(fm, flash, rgb), self.ch, axis=0), self.cw, axis=1)
            fmask = np.repeat(np.repeat(fm, self.ch, axis=0), self.cw, axis=1)
            cell = np.where(fmask, cellf * fade + cell * (1 - fade), cell)
        img[:] = cell
        # gaps: last column and last row of each cell → background
        img[:, self.cw - 1::self.cw] = BG
        img[self.ch - 1::self.ch, :] = BG
        return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))

    def _push_stream(self, s: Step):
        col = cls_color(s.cls)
        text = s.text.replace("\n", " ⏎ ")
        maxw = W - 2 * 96 - 640
        line = self.stream_lines[-1]
        draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        cur = "".join(t for t, _ in line)
        if draw.textlength(cur + text, font=self.f_tok) > maxw:
            self.stream_lines.append([])
            line = self.stream_lines[-1]
        line.append((text, col))
        if len(self.stream_lines) > 4:
            self.stream_lines = self.stream_lines[-4:]

    def frame(self, t_real: float) -> Image.Image:
        # fold in every step whose chunk has arrived
        while self.applied < len(self.steps) and self.steps[self.applied].t <= t_real:
            s = self.steps[self.applied]
            self._apply_step(s)
            self._push_stream(s)
            self.applied += 1
        cur = self.steps[self.applied - 1] if self.applied else None
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        # header
        d.text((96, 22), "DeepSeek-V4-Flash · 2× DGX Spark (TP=2 over RoCE, MTP-5) — what happens behind each token", font=self.f_title, fill=FG)
        elapsed = t_real - self.t0
        d.text((96, 58), f"t = {elapsed:6.2f} s    engine step {self.applied:3d}/{len(self.steps)}    playback ×{1 / self.slowmo:.2g}", font=self.f, fill=DIM)
        # map
        img.paste(self._heat_image(cur, t_real), (self.map_x, self.map_y))
        # node status strips: left edge = dgx01, right edge = dgx02 (red while paging in this step)
        for x, name, paging, anchor, tx in ((self.map_x - 14, "dgx01", cur.paging_head if cur else False, "la", self.map_x - 14),
                                            (self.map_x + self.map_w + 6, "dgx02", cur.paging_worker if cur else False, "ra", W - 6)):
            d.rectangle([x, self.map_y, x + 8, self.map_y + self.map_h], fill=C_PAGING if paging else (50, 56, 66))
            d.text((tx, self.map_y - 26), name, font=self.f_small, fill=C_PAGING if paging else DIM, anchor=anchor)
        d.text((self.map_x + 70, self.map_y - 26), "43 layers × 256 routed experts — one cell per expert; every expert is computed by both nodes (half of its width each), so the map is the same on dgx01 and dgx02", font=self.f_small, fill=DIM)
        for l in range(0, L_LAYERS, 5):
            d.text((self.map_x - 52, self.map_y + l * self.ch - 3), f"L{l:02d}", font=self.f_small, fill=DIM)
        for e in range(0, N_EXPERTS, 32):
            d.text((self.map_x + e * self.cw, self.map_y + self.map_h + 4), f"e{e}", font=self.f_small, fill=DIM)
        # legend
        lx, ly = self.map_x, self.map_y + self.map_h + 26
        for label, colr in (("used → bright (saturates), unused → fades, never used → dark", tuple(int(x) for x in HI)),
                            ("this step: accepted token", C_ACCEPT_FLASH), ("this step: rejected draft only (wasted)", C_REJECT_FLASH),
                            ("edge strip red: that node is paging this step", C_PAGING)):
            d.rectangle([lx, ly + 3, lx + 14, ly + 17], fill=colr)
            d.text((lx + 20, ly), label, font=self.f_small, fill=DIM)
            lx += 30 + int(d.textlength(label, font=self.f_small))
        # ── machine stats (Prometheus 5 s) — narrow row between map and gauges ──
        self._machine_row(d, self.map_y + self.map_h + 72, t_real, cur)
        # ── gauges ──
        gy = self.map_y + self.map_h + 224
        self._gauges(d, gy, cur)
        # ── token stream ──
        sy = H - 250
        d.line([(96, sy - 14), (W - 96, sy - 14)], fill=(40, 46, 56), width=1)
        d.text((96, sy - 40), "generated text (colour = class of the engine step that produced it)", font=self.f_small, fill=DIM)
        for i, line in enumerate(self.stream_lines[-4:]):
            x = 96
            for text, colr in line:
                d.text((x, sy + i * 36), text, font=self.f_tok, fill=colr)
                x += d.textlength(text, font=self.f_tok)
        # newest token: per-layer reuse of the previous token's experts
        if cur is not None and self.last_overlap is not None:
            ex0, ey0 = W - 96 - 600, sy - 40
            shared = sum(self.last_overlap)
            word = self.last_tok_text.strip().replace("\n", " ")[:18]
            slots = cur.rows * L_LAYERS * TOPK
            d.text((ex0, ey0), f"routing continuity \u2014 newest token \u201c{word}\u201d", font=self.f_small, fill=DIM)
            d.text((ex0, ey0 + 22), f"{shared}/{L_LAYERS * TOPK} same experts as previous token (random \u2248 {L_LAYERS * TOPK * TOPK / 256:.0f})", font=self.f, fill=FG)
            bw, bh, by = 12, 60, ey0 + 56
            d.text((ex0, by + bh + 4), "L00", font=self.f_small, fill=DIM)
            d.text((ex0 + (L_LAYERS - 1) * (bw + 1) - 20, by + bh + 4), "L42", font=self.f_small, fill=DIM)
            d.text((ex0 - 30, by - 2), "6", font=self.f_small, fill=DIM)
            d.text((ex0 - 30, by + bh - 14), "0", font=self.f_small, fill=DIM)
            for l, n in enumerate(self.last_overlap):
                x = ex0 + l * (bw + 1)
                d.rectangle([x, by, x + bw - 1, by + bh], fill=(40, 46, 56))
                h = int(bh * n / TOPK)
                if h:
                    d.rectangle([x, by + bh - h, x + bw - 1, by + bh], fill=C_FAST)
            d.text((ex0, by + bh + 22), "per layer (0\u20136): shallow layers re-route, deep layers keep ~half", font=self.f_small, fill=DIM)
            d.text((ex0, by + bh + 40), f"this step: {cur.rows} rows \u2192 {cur.n_distinct} distinct of {slots} slots ({100 * (1 - cur.n_distinct / slots):.0f} % overlap)", font=self.f_small, fill=DIM)
        return img

    MACHINE_COLS = (  # (header, unit, key suffix, format, scale, cumulative-30s?)
        ("RoCE tx", "MB", "ib_tx_bytes", "{:5.1f}", 1e-6, False), ("RoCE rx", "MB", "ib_rx_bytes", "{:5.1f}", 1e-6, False),
        ("tx rate", "Gb/s", "_gbps", "{:4.2f}", 1.0, False),
        ("GPU", "W", "gpu_power_w", "{:4.1f}", 1.0, False), ("util", "%", "gpu_util", "{:3.0f}", 1.0, False),
        ("swap-in", "pages", "swap_in", "{:3.0f}", 1.0, True), ("majfault", "", "majfaults", "{:3.0f}", 1.0, True),
        ("NVMe rd", "KB", "disk_rd_bytes", "{:5.0f}", 1e-3, False),
    )
    SPARK_WINDOW = 30.0  # seconds of history in the mini graphs / cumulative counts

    def _col_value(self, st: Step, prefix: str, key: str, scale: float, host: str, t_real: float):
        if key == "_gbps":
            b = st.raw.get(f"{prefix}_ib_tx_bytes")
            return None if b is None else b * 8 / (st.dur / 1000) / 1e9
        if key == "_temp":
            return self.stats.value_at("gpu_temp_c", host, t_real)
        if key == "_sys_w":
            return self.stats.value_at("system_power_w", host, t_real)
        v = st.raw.get(f"{prefix}_{key}")
        return None if v is None else v * scale

    def _machine_row(self, d: ImageDraw.ImageDraw, y: int, t_real: float, cur: Step | None):
        """dgx01 / dgx02 stacked. Per column: the current engine step's value
        (50 Hz samplers), a 30 s mini graph of that value per step, and for
        swap-in / majfault the 30 s cumulative count next to the step value."""
        if cur is None:
            return
        node_col = {"dgx01": (110, 170, 240), "dgx02": (90, 210, 150)}
        cols = list(self.MACHINE_COLS)
        if self.stats.available and "system_power_w" in self.stats.series:
            gi = next(i for i, c in enumerate(cols) if c[2] == "gpu_power_w")
            cols.insert(gi + 1, ("System", "W", "_sys_w", "{:4.0f}", 1.0, False))
        if self.stats.available and "gpu_temp_c" in self.stats.series:
            cols.append(("temp", "°C", "_temp", "{:3.0f}", 1.0, False))
        d.text((self.map_x, y - 20), f"per node, this engine step (tokentrace samplers, 50 Hz) · mini graph = last {self.SPARK_WINDOW:.0f} s per step · swap-in / majfault: step value + {self.SPARK_WINDOW:.0f} s total", font=self.f_small, fill=DIM)
        # GPU W and System W share one column slot (half each); every other column keeps its width
        weights = [0.5 if c[2] in ("gpu_power_w", "_sys_w") else 1.0 for c in cols]
        unit_w = (self.map_w - 80) / sum(weights)
        xs, x = [], self.map_x + 80
        for wgt in weights:
            xs.append((int(x), int(x + wgt * unit_w)))
            x += wgt * unit_w
        for j, (hdr, unit, *_) in enumerate(cols):
            d.text((xs[j][1] - 8, y), f"{hdr} {unit}".strip(), font=self.f_small, fill=DIM, anchor="ra")
        recent = [st for st in self.steps[: self.applied] if st.t > t_real - self.SPARK_WINDOW]
        row_h = 50
        for i, (host, prefix) in enumerate((("dgx01", "head"), ("dgx02", "worker"))):
            ry = y + 22 + i * row_h
            col = node_col[host]
            d.text((self.map_x, ry + 2), host, font=self.f, fill=col)
            for j, (hdr, unit, key, fmt, scale, cumulative) in enumerate(cols):
                x1 = xs[j][1] - 8
                v = self._col_value(cur, prefix, key, scale, host, t_real)
                hot = key in ("swap_in", "majfaults") and v and v > 0
                txt = "—" if v is None else fmt.format(v)
                if cumulative:
                    tot = sum((self._col_value(st, prefix, key, scale, host, st.t) or 0) for st in recent)
                    txt = f"{txt} / {tot:,.0f}"
                d.text((x1, ry), txt, font=self.f_mid if not cumulative else self.f, fill=C_PAGING if hot else FG, anchor="ra")
                # mini graph: one bar per step over the last 30 s (Prometheus columns: 5 s samples, no graph)
                if key in ("_temp", "_sys_w"):
                    continue
                vals = [self._col_value(st, prefix, key, scale, host, st.t) or 0 for st in recent]
                gx0, gx1, gy1, gh = xs[j][0] + 8, x1, ry + 44, 12
                d.line([(gx0, gy1), (gx1, gy1)], fill=(40, 46, 56))
                if vals:
                    vmax = max(vals) or 1.0
                    n = len(vals)
                    w = (gx1 - gx0) / max(n, 1)
                    for k, val in enumerate(vals):
                        h = int(gh * val / vmax)
                        if h > 0:
                            xa = gx0 + k * w
                            d.rectangle([xa, gy1 - h, xa + max(w - 0.5, 1), gy1], fill=col if not (key in ("swap_in", "majfaults")) else C_PAGING)

    def _gauges(self, d: ImageDraw.ImageDraw, gy: int, cur: Step | None):
        # left: last 120 steps duration bars
        x0, y0, hh = 96, gy, 90
        d.text((x0, y0 - 22), "engine step time (bar = one step, colour = class; line = baseline p50)", font=self.f_small, fill=DIM)
        last = self.steps[max(0, self.applied - 120): self.applied]
        dec = [s.dur for s in self.steps if s.phase == "decode"]
        base = float(np.median(dec)) if dec else 67.0
        scale = hh / (2.0 * base)
        for i, s in enumerate(last):
            h = min(hh, s.dur * scale)
            d.rectangle([x0 + i * 6, y0 + hh - h, x0 + i * 6 + 4, y0 + hh], fill=cls_color(s.cls))
        by = y0 + hh - base * scale
        d.line([(x0, by), (x0 + 120 * 6, by)], fill=(200, 200, 200), width=1)
        d.text((x0 + 120 * 6 + 8, by - 8), f"{base:.0f} ms", font=self.f_small, fill=DIM)
        # middle: current step numbers
        mx = x0 + 120 * 6 + 80
        if cur is None:
            return
        colr = cls_color(cur.cls)
        d.text((mx, y0 - 22), "current step", font=self.f_small, fill=DIM)
        d.text((mx, y0), f"{cur.cls}", font=self.f_big, fill=colr)
        cw_cls = d.textlength("PREFILL", font=self.f_big)
        d.text((mx + cw_cls + 24, y0 + 10), f"{cur.accepted / (cur.dur / 1000):5.1f} tok/s", font=self.f_mid, fill=colr)
        d.text((mx, y0 + 48), f"{cur.dur:6.1f} ms   {cur.accepted}/{cur.rows} accepted", font=self.f, fill=FG)
        # accepted/rejected boxes
        bx = mx
        for i in range(cur.rows):
            fill = C_FAST if i < cur.accepted else (70, 70, 80)
            d.rectangle([bx + i * 26, y0 + 76, bx + i * 26 + 20, y0 + 94], fill=fill, outline=(90, 90, 100))
        d.text((mx, y0 + 98), "rows: green = accepted, grey = rejected", font=self.f_small, fill=DIM)
        # right: fabric / power / paging / coverage
        rx = 1300
        d.text((rx, y0 - 22), "experts", font=self.f_small, fill=DIM)
        cov = int(self.ever.sum())
        d.text((rx, y0), f"touched this step  {cur.n_distinct:4d} / 11008  ({100 * cur.n_distinct / 11008:4.1f} %)", font=self.f, fill=FG)
        d.text((rx, y0 + 28), f"used so far        {cov:5d} / 11008  ({100 * cov / 11008:4.1f} %)", font=self.f, fill=FG)
        d.rectangle([rx, y0 + 60, rx + 560, y0 + 68], outline=(80, 86, 96))
        d.rectangle([rx, y0 + 60, rx + int(560 * cov / 11008), y0 + 68], fill=tuple(int(x) for x in HI))
        d.text((rx, y0 + 76), "11008 = 43 layers × 256 experts", font=self.f_small, fill=DIM)
        d.text((rx, y0 + 94), "each (layer, expert) has its own weights → counted per pair", font=self.f_small, fill=DIM)
        d.text((rx, y0 + 112), "an expert index shares nothing across layers", font=self.f_small, fill=DIM)


def _hsv(h, s, v):
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return int(r * 255), int(g * 255), int(b * 255)


def render(steps: list[Step], out: str, slowmo: float, fps: int, tau: float, limit_s: float | None = None, flash_frames: float = 4.0,
           stats: "MachineStats | None" = None):
    R = Renderer(steps, slowmo, fps, tau, flash_frames, stats)
    dur_real = (R.t_end - R.t0)
    if limit_s:
        dur_real = min(dur_real, limit_s)
    n_frames = int(dur_real * slowmo * fps)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n_frames):
        t_real = R.t0 + i / (fps * slowmo)
        img = R.frame(t_real)
        p.stdin.write(img.tobytes())
        if i % 300 == 0:
            print(f"  frame {i}/{n_frames}", file=sys.stderr, flush=True)
    p.stdin.close()
    p.wait()
    return n_frames


def build_parser(p=None):
    p = p or argparse.ArgumentParser(prog="tokentrace video")
    p.add_argument("--steps", required=True)
    p.add_argument("--chunks", required=True)
    p.add_argument("--experts", required=True, help="host dir with experts-*.idx.jsonl")
    p.add_argument("--req", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--slowmo", type=float, default=1.0, help="4 = quarter-speed playback")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--tau", type=float, default=4.0, help="heat decay time constant (s of generation time)")
    p.add_argument("--limit", type=float, default=None, help="only the first N real seconds")
    p.add_argument("--flash-frames", type=float, default=4.0, help="frames a step flash stays visible (decaying)")
    p.add_argument("--grafana", default=None, help="Prometheus query_range dump (gpu_temp_c / gpu_power_w / rdma_rx_gbps per host)")
    p.add_argument("--still", type=float, default=None, help="also write one PNG at this real second")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    steps = build(a.steps, a.chunks, a.experts, a.req)
    stats = MachineStats(a.grafana)
    print(f"steps {len(steps)} classes {dict((c, sum(1 for s in steps if s.cls == c)) for c in ('PREFILL', 'FAST', 'NORMAL', 'STALL'))}")
    if a.still is not None:
        R = Renderer(steps, a.slowmo, a.fps, a.tau, a.flash_frames, stats)
        R.frame(R.t0 + a.still).save(a.out.rsplit(".", 1)[0] + f"-{a.still:.1f}s.png")
    n = render(steps, a.out, a.slowmo, a.fps, a.tau, a.limit, a.flash_frames, stats)
    print(f"wrote {a.out}: {n} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
