"""tokentrace.video — data model, heat/flash logic, frame rendering and a
short ffmpeg render over synthetic inputs (pytest; skips without Pillow /
ffmpeg)."""
import json
import shutil

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")

from test_tokentrace_runtime import write_expert_log  # noqa: E402
from tokentrace import video  # noqa: E402

L, K = video.L_LAYERS, video.TOPK


def _synthetic(tmp_path, n_steps=6):
    """analyze-style steps JSON + chunks JSON + recorder log for one request."""
    rng = np.random.default_rng(0)
    t0 = 2000.0
    rows, chunks, log_steps = [], [], []
    for k in range(n_steps):
        dur = 800.0 if k == 0 else (67.0 if k % 3 else 90.0)
        t_end = t0 + sum((800.0 if j == 0 else (67.0 if j % 3 else 90.0)) for j in range(k + 1)) / 1000
        rows.append({"i": k, "req": "tt-x", "k": k, "phase": "prefill" if k == 0 else "decode", "t_start": t_end - dur / 1000,
                     "t_end": t_end, "dur_ms": dur, "head_ib_tx_bytes": 12_600_000, "head_ib_rx_bytes": 12_600_000,
                     "head_ib_pkts": 18_300, "head_gpu_power_w": 24.5, "worker_gpu_power_w": 25.0,
                     "paging_flags": ["worker-swap-in"] if k == 2 else []})
        chunks.append({"i": k, "t": t_end, "text": f"tok{k} ", "tokens": [k]})
        n = 8 if k == 0 else 6
        sampled = 8 if k == 0 else (6 if k == 1 else 1 + k % 5)
        exp_rows = [[list(map(int, rng.choice(256, K, replace=False))) for _ in range(L)] for _ in range(n)]
        log_steps.append((n, sampled, exp_rows))
    steps_json = tmp_path / "steps.json"
    steps_json.write_text(json.dumps({"summary": {}, "steps": rows}))
    chunks_json = tmp_path / "chunks.json"
    chunks_json.write_text(json.dumps(chunks))
    idx, _ = write_expert_log(tmp_path / "dgx01", log_steps, L=L, K=K, t0=t0 + 0.8)
    # recorder timestamps must fall inside the request window: rewrite t per step
    lines = idx.read_text().splitlines()
    fixed = [lines[0]]
    for k, line in enumerate(lines[1:]):
        rec = json.loads(line)
        rec["t"] = rows[k]["t_end"]
        fixed.append(json.dumps(rec))
    idx.write_text("\n".join(fixed) + "\n")
    return steps_json, chunks_json, tmp_path / "dgx01", log_steps


def test_build_aligns_steps_and_classifies(tmp_path):
    sj, cj, ed, log_steps = _synthetic(tmp_path)
    steps = video.build(str(sj), str(cj), str(ed), "tt-x")
    assert len(steps) == 6
    assert steps[0].cls == "PREFILL" and steps[0].rows == 8 and steps[0].accepted == 8
    assert steps[1].cls == "FAST"          # 6/6 accepted at baseline
    assert steps[3].cls == "STALL"         # 90 ms > 1.15 × 67
    assert steps[2].paging_worker and not steps[2].paging_head
    assert steps[1].experts.shape == (6, L, K) and steps[1].experts[0, 0].tolist() == log_steps[1][2][0][0]
    assert steps[1].n_distinct <= 6 * L * K and steps[1].ib_mb == pytest.approx(25.2)
    assert steps[4].text == "tok4 "


def test_heat_saturates_decays_and_flashes(tmp_path):
    sj, cj, ed, _ = _synthetic(tmp_path)
    steps = video.build(str(sj), str(cj), str(ed), "tt-x")
    R = video.Renderer(steps, slowmo=1.0, fps=10, tau=2.0)
    s1 = steps[1]
    R._apply_step(steps[0])
    R._apply_step(s1)
    l, e = 0, int(s1.experts[0, 0, 0])
    assert R.ever[l, e] and 0.2 < R.heat[l, e] <= 1.0
    before = R.heat.copy()
    later = video.Step()
    later.t, later.accepted, later.experts = s1.t + 4.0, 0, np.zeros((0, L, K), dtype=np.uint8)
    R._apply_step(later)
    assert np.all(R.heat <= before) and R.heat[l, e] == pytest.approx(before[l, e] * np.exp(-2.0), rel=1e-3)
    assert R.prev_tok is not None and len(R.last_overlap) == L and all(0 <= n <= K for n in R.last_overlap)
    assert R.last_tok_text == steps[1].text
    acc, rej = R._current_sets(steps[2])  # accepted 3 of 6 rows
    assert acc.any() and not (acc & rej).any()
    img = R._heat_image(steps[2], steps[2].t)
    assert img.size == (R.map_w, R.map_h)


def test_flash_decays_over_frames(tmp_path):
    sj, cj, ed, _ = _synthetic(tmp_path)
    steps = video.build(str(sj), str(cj), str(ed), "tt-x")
    R = video.Renderer(steps, slowmo=1.0, fps=30, tau=2.0, flash_frames=4)
    R.frame(steps[1].t)  # step 1 just landed
    acc, _ = R._current_sets(steps[1])
    l, e = next(zip(*np.nonzero(acc)))
    px = (l * R.ch + 2, e * R.cw + 1)
    at0 = np.asarray(R._heat_image(steps[1], steps[1].t))[px].astype(int)
    at2 = np.asarray(R._heat_image(steps[1], steps[1].t + 2 / 30))[px].astype(int)
    at9 = np.asarray(R._heat_image(steps[1], steps[1].t + 9 / 30))[px].astype(int)
    assert tuple(at0) == video.C_ACCEPT_FLASH                      # frame 0: full white
    assert at0.sum() > at2.sum() > at9.sum()                       # decaying
    assert R.flash_tau == pytest.approx((4 / 30) / 2.3)


def test_frame_and_stream_scroll(tmp_path):
    sj, cj, ed, _ = _synthetic(tmp_path)
    steps = video.build(str(sj), str(cj), str(ed), "tt-x")
    R = video.Renderer(steps, slowmo=1.0, fps=10, tau=2.0)
    img = R.frame(R.t0)  # before any chunk
    assert img.size == (video.W, video.H) and R.applied == 0
    img = R.frame(steps[-1].t + 0.1)
    assert R.applied == 6 and img.size == (video.W, video.H)
    assert "".join(t for line in R.stream_lines for t, _ in line).startswith("tok0 tok1")
    assert len(R.stream_lines) <= 4
    assert video.cls_color("STALL") == video.C_STALL and video._hsv(0.0, 1, 1) == (255, 0, 0)


def test_machine_stats_interpolation_and_row(tmp_path):
    g = tmp_path / "g.json"
    g.write_text(json.dumps({"series": {"gpu_temp_c": {"dgx01": [[2000.0, 50], [2010.0, 60]], "dgx02": [[2000.0, 52], [2010.0, 52]]},
                                        "gpu_power_w": {"dgx01": [[2000.0, 10], [2010.0, 25]]},
                                        "rdma_rx_gbps": {"dgx01": [[2000.0, 0.0], [2005.0, 3.0], [2010.0, 0.0]]}}}))
    ms = video.MachineStats(str(g))
    assert ms.available and ms.hosts("gpu_temp_c") == ["dgx01", "dgx02"]
    assert ms.value_at("gpu_temp_c", "dgx01", 2005.0) == 50          # step-hold, not interpolated
    assert ms.value_at("gpu_temp_c", "dgx01", 2010.0) == 60
    assert ms.value_at("gpu_temp_c", "dgx01", 1990.0) == 50 and ms.value_at("gpu_temp_c", "dgx01", 2020.0) == 60
    assert ms.changed_at("gpu_temp_c", "dgx01", 2012.0) == 2010.0 and ms.changed_at("gpu_temp_c", "nope", 1) is None
    assert ms.value_at("gpu_temp_c", "nope", 2005.0) is None
    assert ms.bounds("gpu_temp_c", 2000.0, 2010.0) == (50, 60)
    assert not video.MachineStats(None).available
    sj, cj, ed, _ = _synthetic(tmp_path)
    steps = video.build(str(sj), str(cj), str(ed), "tt-x")
    R = video.Renderer(steps, slowmo=1.0, fps=10, tau=2.0, stats=ms)
    img = R.frame(steps[2].t)
    assert img.size == (video.W, video.H)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_render_short_clip(tmp_path):
    sj, cj, ed, _ = _synthetic(tmp_path)
    out = tmp_path / "clip.mp4"
    rc = video.main(["--steps", str(sj), "--chunks", str(cj), "--experts", str(ed), "--req", "tt-x", "--out", str(out),
                     "--slowmo", "1", "--fps", "5", "--limit", "1.0", "--still", "0.9"])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 1000
    assert (tmp_path / "clip-0.9s.png").exists()
