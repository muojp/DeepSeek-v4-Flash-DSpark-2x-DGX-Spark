# tokentrace — end-to-end runs (2026-08-28)

Two 1000-token runs of the same prompt (86 prompt tokens, `ignore_eos`,
`thinking=false`) on the live pair. Run 1 against the 6-day-old server
(process-external telemetry only); run 2 right after a restart with the
V2-runner expert recorder (`DSPARK_TOKENTRACE_EXPERTS=1`), which also
captured the model load. Method and schema: [`docs/TOKENTRACE.md`](../docs/TOKENTRACE.md).
Per-step tables: [`tokentrace-e2e-2026-08-28-steps.json`](tokentrace-e2e-2026-08-28-steps.json) (run 1),
[`tokentrace-e2e-2026-08-28-run2-steps.json`](tokentrace-e2e-2026-08-28-run2-steps.json) (run 2).

Stack: anemll 0.1.1, TP=2 over RoCE (rocep1s0f0, 200 Gb/s), MTP-5 (dspark),
V2 model runner, gb10-clock-cap active (2177 MHz) on both nodes. Samplers at
50 Hz while a request is active; vLLM `/metrics` at 10 Hz; worker clock
aligned by the samplers' UDP exchange (offset −67 ms → −53 ms after the
reboot of the pair, fabric RTT 24 µs).

## Headline (run 1 → run 2)

| | run 1 (old server) | run 2 (fresh, recorder on) |
|---|---|---|
| decode rate (client) | 46.6 tok/s | 45.2 tok/s |
| engine step p50 / p90 | 66.9 / 71.1 ms | 67.7 / 77.6 ms |
| accepted tokens / step | 3.11 (vLLM counter) | **3.08 exact** (recorder: hist 1:61 2:68 3:59 4:33 5:33 6:38) |
| rows verified / step | — | **6.27** (1 + 5 drafts, occasionally more) |
| fabric per step, each direction, each node | 12.6 MB / 18.3 k pkts (688 B/pkt) | 12.6 MB / 18.3 k pkts |
| fabric busy fraction (20 ms bins) | 1.00 | 1.00 |
| link utilisation | 0.75 % of 200 Gb/s | 0.75 % |
| GPU power idle / prefill / decode | 9.7 / 13.5 / 24.5 W | 9.7 / 10.9 / 24.5 W |
| head NVMe + swap during decode | 0 | 13 KB/step, 2.7 pages/step (cold page cache after reboot) |
| worker NVMe + swap during decode | 15 KB/step, 3.8 pages/step | 108 KB/step, 26 pages/step |
| paging cost (Δ step time, with vs without) | worker +1.5 ms | head −1.2 ms, worker +1.0 ms → noise-level |
| TTFT, 86-token prompt | **1.63 s** | **0.55 s** |
| step time vs accepted tokens | flat 66–69 ms | flat 62–69 ms |

## Expert routing (run 2, from the runner log)

- Every step verifies 6.27 rows (bonus token + drafts). Rows touch
  **937 distinct (layer, expert) pairs per step on average (8.5 % of
  43 × 256), max 4940 in the prefill step**.
- Over one 1000-token request **9 482 of 11 008 (layer, expert) pairs
  (86 %) are used at least once** — the whole expert set must be resident;
  a paging scheme would fault continuously.
- Per real token 258 = 43 × 6 experts (no duplicates within a layer, as
  expected); between consecutive real tokens **64 % of the expert slots
  change** (36 % are shared).
- Routing is well spread: normalised entropy per layer 0.71–0.9; the least
  spread layers are 13, 8, 26 (180–184 distinct experts each).
- Weight traffic estimate (not measured — GB10 exposes no memory-bandwidth
  counter): one expert is 3 × 4096 × 2048 params in mxfp4 ≈ 13 MB, split
  across TP=2 → ~6.7 MB per node. 937 experts/step ≈ **6 GB of expert
  weights per node per step**, i.e. ~90 GB/s of the 273 GB/s LPDDR5X
  (≈ 22 ms of the 67 ms step at full bandwidth). The prefill step's 4 940
  experts ≈ 33 GB. So decode is a chain of small fabric round trips *and*
  a sizeable weight stream from unified memory; neither the link nor the
  SMs are the limiter (24.5 W).

## Model load (run 2, model_load_start → model_ready, 413 s)

| | dgx01 | dgx02 |
|---|---|---|
| NVMe read | 143.7 GB | 146.7 GB |
| NVMe written | 3.3 GB | 9.8 GB |
| RoCE tx | 13.1 GB | 13.2 GB |
| swap-in | 346 k pages (1.4 GB) | 653 k pages (2.7 GB) |
| major faults | 273 k | 691 k |
| GPU power / util (mean) | 10.7 W / 7.5 % | 10.6 W / 7.8 % |

Each node reads essentially the full 155 GiB checkpoint (TP slices after
load), converts on the CPU (GPU nearly idle the whole time), and exchanges
13 GB over the fabric (NCCL init, CUDA-graph capture, warm-up). That is the
pre-paid cost: after `model_ready` the serving phase reads ~0 from NVMe on
the head. Both nodes swap during the load ("Available RAM: 27 GiB" at load
time) — the second thing to fix after the worker's steady-state pressure.
The 10-second timeline for both nodes is in the run-2 JSON
(`summary.load_phase.<host>.timeline_10s` and `mem_ramp`).

## What this says about the hypothesis

1. **Experts are resident; no reload.** Steady-state NVMe/swap traffic is
   KB-per-step background paging on a memory-tight box, three orders of
   magnitude below the ~6 GB/step of expert weights a step actually
   touches. 86 % of all experts are used within one request.
2. **Inter-node traffic is small and latency-shaped.** 12.6 MB/step in
   18 k packets (~700 B), present in every 20 ms bin: per-layer
   collectives, not bulk transfer. 0.75 % link utilisation.
3. **The step cost is fixed; MTP tokens are free.** 62–69 ms regardless
   of how many drafts were accepted; 3.08 accepted/step × 14.8 steps/s
   ≈ 45 tok/s. Acceptance (0.42 on this prompt vs 0.69 fleet average) is
   the lever, not the step.
4. **Where the 67 ms goes (best current split):** ≥ 22 ms streaming
   ~6 GB of expert weights from LPDDR5X per node, the rest the
   43-layer chain of RoCE round trips plus attention/indexer over a growing
   context; SMs are mostly waiting (24.5 W, 95 % "util" from spin
   kernels). Splitting this further needs an in-process profile (torch
   profiler / nsys) at a later restart.
5. **Memory pressure is the standing risk.** Both nodes swap after the
   load; the worker keeps faulting during decode. Cost today ≈ 1 ms/step;
   it is the first thing that becomes a cliff under concurrency.
6. **TTFT for tiny prompts varies 3×** between a cold 6-day-old process
   (1.63 s) and a fresh one (0.55 s) with identical fabric traffic
   (86 MB); the difference sits on the CPU/paging side, not in the GPU.

## Caveats

- One prompt, one stream; the c=2…6 and long-prompt matrix is next.
- vLLM per-step token counts at 10 Hz are aliased (use the recorder's
  `sampled` instead); NVML `utilization.memory` is 0 on GB10.
- Weight-bytes-per-step is an estimate from the routing log and the
  checkpoint layout, not a measurement.

## Raw report, run 2 (tokentrace analyze)

# tokentrace report

steps: 328 (prefill 1, decode 327)
clock offset worker−head: -53358 µs (fabric RTT 24 µs)

- request `tt-20260827-182137-ceab4b`: chunks=328 TTFC=0.545 s stream=22.1261 s decode=45.15 tok/s usage={'prompt_tokens': 86, 'total_tokens': 1086, 'completion_tokens': 1000, 'prompt_tokens_details': {'cached_tokens': 0}}

## Decode steps (1 chunk = 1 engine step)

| metric | mean | p50 | p90 | max |
|---|---:|---:|---:|---:|
| dur_ms | 67.66 | 67.66 | 77.62 | 83.68 |
| engine_steps | 0.99 | 1 | 2 | 2 |
| gen_tokens | 3.04 | 3 | 7 | 12 |
| head_ib_tx_bytes | 12607103.23 | 12609654 | 14690769 | 15767982 |
| head_ib_rx_bytes | 12611946.83 | 12575637 | 14653213 | 15790028 |
| head_ib_pkts | 18338.13 | 18333 | 21503 | 22778 |
| head_ib_busy_frac | 1.0 | 1.0 | 1.0 | 1.0 |
| head_gpu_util | 93.4 | 95.0 | 95.0 | 95.0 |
| head_gpu_mem_util | 0.0 | 0.0 | 0.0 | 0.0 |
| head_gpu_power_w | 24.5 | 24.9 | 25.1 | 25.3 |
| head_disk_rd_bytes | 13107.56 | 7426 | 25680 | 687935 |
| head_swap_in | 2.71 | 1 | 7 | 26 |
| head_majfaults | 1.61 | 1 | 4 | 14 |
| worker_ib_tx_bytes | 12592193.97 | 12573196 | 14314446 | 15539559 |
| worker_gpu_util | 94.11 | 95.0 | 96.0 | 96.0 |
| worker_gpu_mem_util | 0.0 | 0.0 | 0.0 | 0.0 |
| worker_disk_rd_bytes | 108424.83 | 0 | 420767 | 1060815 |
| worker_swap_in | 26.07 | 0 | 99 | 252 |
| worker_majfaults | 24.6 | 0 | 96 | 237 |
| client_gap_ms | 0.0 | 0.0 | 0.0 | 0.0 |
| accept_ratio | 0.42 | 0.4 | 0.8 | 1.0 |

decode tok/s (vLLM counters over step time): **44.92**; fabric tx 186.32 MB/s, 4147407 B/token, 687.5 B/packet
wait classes: {'fabric-latency': 327}  (NVML util counts NCCL spin kernels as busy — the fabric-busy fraction and GPU power are the discriminators)
paging flags: {'head-swap-in': 227, 'head-majfault': 222, 'worker-swap-in': 125, 'worker-majfault': 125, 'head-disk-read': 5, 'worker-disk-read': 102}
- head paging cost: 228 steps with paging avg 67.31 ms vs 99 without avg 68.48 ms → Δ -1.17 ms
- worker paging cost: 145 steps with paging avg 68.22 ms vs 182 without avg 67.22 ms → Δ 1.0 ms
- GPU power (W): {'head': {'decode': 24.5, 'prefill': 10.9}, 'worker': {'decode': 25.1, 'prefill': 9.8}}
- step time by accepted tokens (vLLM counter, aliased by the 10 Hz poll): 0→68.28ms(n=107), 1→66.71ms(n=21), 2→69.02ms(n=33), 3→68.83ms(n=35), 4→67.67ms(n=25), 5→65.6ms(n=35), 6→69.24ms(n=36), 7→64.81ms(n=9), 8→62.4ms(n=9), 9→67.72ms(n=3), 10→65.31ms(n=8), 11→62.72ms(n=3), 12→64.67ms(n=3)

## Prefill (send → first chunk)

- dur_ms: {'mean': 548.41, 'p50': 548.41, 'p90': 548.41, 'max': 548.41, 'sum': 548.41}
- engine_steps: {'mean': 0, 'p50': 0, 'p90': 0, 'max': 0, 'sum': 0}
- head_ib_tx_bytes: {'mean': 86212136, 'p50': 86212136, 'p90': 86212136, 'max': 86212136, 'sum': 86212136}
- head_gpu_util: {'mean': 0.0, 'p50': 0.0, 'p90': 0.0, 'max': 0.0, 'sum': 0.0}
- head_disk_rd_bytes: {'mean': 2160660, 'p50': 2160660, 'p90': 2160660, 'max': 2160660, 'sum': 2160660}
- worker_disk_rd_bytes: {'mean': 7497103, 'p50': 7497103, 'p90': 7497103, 'max': 7497103, 'sum': 7497103}

## Expert routing (runner log, this request)

- steps 292, real tokens 899, rows/step 6.27 (1 + drafts), accepted/step 3.08 hist {1: 61, 3: 59, 4: 33, 2: 68, 5: 33, 6: 38}
- distinct (layer, expert) pairs touched per step: mean 936.6, max 4940 of 11008 (8.5 %)
- over the whole request: 9482 distinct (layer, expert) pairs (86.1 % of all experts)
- per real token: 258 experts (max 258); slots changing token→token: 0.644
- least spread layers (normalised entropy): [(13, 184, 0.709), (8, 182, 0.741), (26, 180, 0.742)]

## Model load (model_load_start → model_ready)

- duration 412.8 s
- dgx01: NVMe read 143.7 GB, written 3.3 GB, IB tx 13.077 GB, TCP tx 0.003 GB, swap-in 346302 pages, majfaults 272559, GPU 10.7 W / util 7.5 %
- dgx02: NVMe read 146.65 GB, written 9.84 GB, IB tx 13.235 GB, TCP tx 0.003 GB, swap-in 653197 pages, majfaults 691430, GPU 10.6 W / util 7.8 %

errors by source: {'dgx01': Counter({'vllm': 7, 'clock': 1}), 'dgx02': Counter()}

