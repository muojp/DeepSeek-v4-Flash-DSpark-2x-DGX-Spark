# tokentrace — what happens behind every token

Per-token telemetry for the 2× DGX Spark DSpark stack: what the unified
memory, the ConnectX-7 fabric, the NVMe and the GPU are doing while a
request is in prefill, decode, and while the model is being loaded.

The goal is to *explain* the ~50 tok/s decode rate — which resource each
engine step is waiting on — and to test the working hypothesis:

> All 256 routed experts of every layer are resident in unified memory, so
> expert switching costs nothing (no reload, no SSD, no host↔device copy);
> the two nodes only exchange the small per-layer activations, so decode is
> latency-bound on the fabric, not bandwidth-bound.

## 0. Ground truth from the pilot (2026-08-28, dgx01, 200-token request)

| Phase | IB `rocep1s0f0` tx / rx | IB pkts/s | `/proc/net/dev` tx / rx |
|---|---|---|---|
| idle | 0 / 0 MB/s | 0 | 0.3 / 0.4 KB/s |
| prefill (3.35 s) | 5.4 / 5.4 MB/s | 7 k | 1.0 / 0.5 KB/s |
| decode (6.7 s) | **170.6 / 170.7 MB/s** | **248 k** | 13.2 / 1.4 KB/s |

- 90 engine steps in 6.66 s → **74 ms/step**, 2.2 accepted tokens/step (MTP-5),
  **12.8 MB per step** over RoCE, present in 94 % of 10 ms bins (continuous,
  not bursty). 171 MB/s is **0.7 %** of the 200 Gb/s link.
- The "~100 KB/s on the QSFP link" seen on the dashboard is the TCP side
  (`/proc/net/dev`). NCCL uses `NCCL_NET=IB` → RDMA verbs bypass the kernel
  network stack; the bytes only appear in
  `/sys/class/infiniband/<hca>/ports/1/counters/port_{xmit,rcv}_data` (×4).
- NVML on GB10: `nvmlDeviceGetUtilizationRates`, power, SM clock and
  `nvmlDeviceGetComputeRunningProcesses` work (the vLLM worker shows
  107.9 GB); `GetMemoryInfo` and PCIe counters are *Not Supported*
  (unified memory). The util/mem-util sample ring buffer ticks every 200 ms.
- Both nodes are swapping a little (dgx01 SwapCached 1.27 GB, MemAvailable
  8.8 GB; dgx02 MemAvailable 2.8 GB). Swap-in during decode would be a real
  "wait" and must be captured (`/proc/vmstat pswpin`).

## 1. Sources and what each answers

| Question | Source | Cadence | Cost |
|---|---|---|---|
| When does each engine step start/end? | vLLM `/metrics`: `vllm:iteration_tokens_total_count` (step counter), `_sum` (tokens per step), `generation_tokens_total`, `spec_decode_num_{drafts,draft_tokens,accepted_tokens}_total`, `spec_decode_num_accepted_tokens_per_pos_total`, `num_requests_{running,waiting}`, `kv_cache_usage_perc`, `prompt_tokens_by_source_total` | 10 Hz (3 ms/67 KB per call) | ~3 % of a core |
| When does the client see each token? | probe: SSE chunk arrival timestamps (monotonic + wall), `usage` | per chunk | — |
| How much crosses the fabric, and when? | sysfs IB counters `port_{xmit,rcv}_data` (×4 B), `port_{xmit,rcv}_packets`, `port_xmit_wait`, hw_counters `rx_write_requests`, `out_of_sequence`, `packet_seq_err`, `local_ack_timeout_err`, `np_cnp_sent`, `rp_cnp_handled` | 50 Hz while a request is active, 1 Hz idle | ~1 % |
| Kernel-side TCP (gloo/control plane) | `/sys/class/net/<if>/statistics/{rx,tx}_bytes` | same | — |
| Is the GPU busy, and how hard is memory being hit? | NVML util.gpu / util.memory (instantaneous), power, SM clock; 200 ms sample ring for device-timestamped util/mem-util | 50 Hz instantaneous, ring drained at 1 Hz | <1 % |
| Does the SSD move at all during decode? (expert paging, swap) | `/proc/diskstats nvme0n1` sectors read/written, io ticks; `/proc/vmstat` `pswpin`, `pswpout`, `pgmajfault` | same | — |
| Unified memory pressure | `/proc/meminfo` MemFree/MemAvailable/Cached/AnonPages/Shmem/SwapFree; NVML compute-process memory (GPU-held part of the pool) | 1 Hz | — |
| CPU-side stalls (scheduler, tokenizer, Python) | `/proc/stat` cpu totals; `/proc/<pid>/stat` utime/stime/`voluntary_ctxt_switches` for `VLLM::EngineCore` and `VLLM::Worker_TP*` | 10 Hz | — |
| Which experts fired for which token? | **runner-side log** (`patches/hotfix-dsv4-tokentrace-experts.py`, gate `DSPARK_TOKENTRACE_EXPERTS=1`): the V2 GPU model runner copies every MoE router's `topk_ids` into a fixed device buffer inside the CUDA graph and appends `[n, 43, 6]` uint8 per engine step + an index line (`req`, `sched`, `pos`, `draft`, `sampled`, `rejected`) to `~/.cache/huggingface/tokentrace/experts-*.{idx.jsonl,u8}` on each node. `--enable-return-routed-experts` is unusable: DSpark exists only in the V2 runner, which rejects it. | per step | 39 µs/step, 258 B/token; needs a restart to enable |
| Cross-node clock offset | UDP 4-timestamp exchange between the two samplers over the fabric | 1 Hz | — |

Not available without a restart, and out of scope for now: in-step kernel
breakdown (torch profiler / nsys), NCCL flight recorder. The schema leaves
room for them (`profile` records).

## 2. Data structures

One append-only JSONL file per node per day:

```
~/tokentrace/<hostname>/trace-YYYYMMDD.jsonl     (sampler)
~/tokentrace/<hostname>/probe-YYYYMMDD.jsonl     (probe, head node only)
```

Every record has `t` (wall clock, `time.time()`), `m` (monotonic seconds
of the writing process), `host`, and `type`. Counters are stored **raw and
cumulative**; deltas are computed at analysis time so a dropped sample never
corrupts the series.

### 2.1 `meta` — once per sampler start

```json
{"type":"meta","t":…,"m":…,"host":"dgx01","boot_id":"a72d…","kernel":"6.17.0-1029-nvidia",
 "role":"head","pid":1234,"version":1,
 "config":{"fast_hz":50,"slow_hz":1,"vllm_hz":10,"vllm_url":"http://127.0.0.1:8888","peer":"192.168.100.40"},
 "hca":[{"name":"rocep1s0f0","port":1,"rate":"200 Gb/sec (2X NDR)","state":"ACTIVE"}],
 "net":["enp1s0f0np0"],"disks":["nvme0n1"],
 "gpu":{"name":"NVIDIA GB10","nvml":"580.173","supports":{"util":true,"memory_info":false,"pcie":false,"samples":["gpu_util","mem_util"]}},
 "columns":{"fast":["ib_rx_bytes","ib_tx_bytes","ib_rx_pkts","ib_tx_pkts","ib_xmit_wait","net_rx_bytes","net_tx_bytes","gpu_util","gpu_mem_util","gpu_power_mw","gpu_sm_mhz","disk_rd_sectors","disk_wr_sectors","disk_io_ticks","pswpin","pswpout","pgmajfault"]}}
```

`columns.fast` declares the layout of the compact `fast` rows, so the format
can evolve without breaking readers.

### 2.2 `fast` — 50 Hz while armed, 1 Hz idle (compact row)

```json
{"type":"fast","t":1787852246.3168,"m":12345.678,"v":[238774510192,238067359390,1231952504,1201990809,0,268013599,626398482,0,0,9650,2177,847031030,485704138,2184855,0,0,0]}
```

Column order is `meta.columns.fast`. Only the *first* HCA port that is
ACTIVE and named in `NCCL_IB_HCA` is in the fast row; all ports are in
`slow`.

### 2.3 `slow` — 1 Hz, full detail

```json
{"type":"slow","t":…,"m":…,
 "ib":{"rocep1s0f0/1":{"rx_bytes":…,"tx_bytes":…,"rx_pkts":…,"tx_pkts":…,"xmit_wait":0,"rx_write_req":…,"out_of_sequence":0,"packet_seq_err":0,"local_ack_timeout_err":0,"np_cnp_sent":0,"rp_cnp_handled":0}},
 "net":{"enp1s0f0np0":{"rx_bytes":…,"tx_bytes":…}},
 "mem":{"MemTotal":…,"MemFree":…,"MemAvailable":…,"Cached":…,"AnonPages":…,"Shmem":…,"SwapTotal":…,"SwapFree":…,"SwapCached":…,"Dirty":…,"Mapped":…},
 "vmstat":{"pswpin":…,"pswpout":…,"pgmajfault":…,"pgfault":…},
 "disk":{"nvme0n1":{"rd_ios":…,"rd_sectors":…,"wr_ios":…,"wr_sectors":…,"io_ticks":…,"in_flight":0}},
 "cpu":{"user":…,"nice":…,"system":…,"idle":…,"iowait":…,"irq":…,"softirq":…},
 "procs":{"306134":{"comm":"VLLM::Worker_TP0","utime":…,"stime":…,"rss_kb":…,"vcsw":…,"nvcsw":…},"306031":{"comm":"VLLM::EngineCore",…}},
 "gpu":{"util":0,"mem_util":0,"power_mw":9650,"sm_mhz":2177,"procs":[{"pid":306134,"mem_bytes":107933581312}],
        "samples":{"gpu_util":[[1787852246.10,0],[1787852246.30,0]],"mem_util":[[…]]}}}
```

`gpu.samples` are the NVML ring-buffer samples newer than the last drain,
with the **device** timestamp converted to wall time (`t_wall = t_now -
(now_us - sample_us)/1e6`), so the 200 ms GPU/memory-controller
utilisation windows line up with everything else.

### 2.4 `vllm` — 10 Hz (head node)

```json
{"type":"vllm","t":…,"m":…,"ok":true,"latency_ms":2.7,
 "steps":123456,"step_tokens":987654,"gen_tokens":128350,"prompt_tokens":…,
 "prompt_cached":…,"prompt_local_compute":…,
 "drafts":28769,"draft_tokens":143845,"accepted_tokens":99586,"accepted_per_pos":[26167,22962,19768,16766,13923],
 "running":1,"waiting":0,"kv_cache_usage":0.0012,"preemptions":0}
```

`steps` is `vllm:iteration_tokens_total_count`; a step boundary is any
sample where it increments. Consecutive vllm samples bracket each step to
±100 ms; the fast rows inside the bracket give the fabric/GPU/disk activity
per step; the probe's chunk timestamps pin the step to ±1 ms when a request
is streaming (one chunk per step in practice — see §4).

### 2.5 `clock` — 1 Hz, head node (offset of the peer's clock)

```json
{"type":"clock","t":…,"m":…,"peer":"192.168.100.40","rtt_us":83,"offset_us":-412,"n":8}
```

NTP-style: `offset = ((t1 - t0) + (t2 - t3)) / 2` over the fabric (RTT
≈ 100 µs), median of `n` exchanges. Analysis shifts the worker's `t` by
`-offset`. Both nodes run systemd-timesyncd, but ±10 ms is not good enough
for 74 ms steps; the UDP exchange gives sub-millisecond alignment.

### 2.6 `mark` — free-form phase markers

```json
{"type":"mark","t":…,"m":…,"label":"model_load_start","note":"docker compose up","source":"cli"}
```

Written by `tokentrace mark <label>` (or by the probe at request start/end).
Model load visualisation (§5) is bracketed by `model_load_start` /
`model_ready` marks plus the vLLM `/health` transition.

### 2.7 `err` — a source failed (never fatal)

```json
{"type":"err","t":…,"m":…,"source":"nvml","error":"NVML_ERROR_NOT_SUPPORTED","count":3}
```

Each source has its own error budget: after 10 consecutive failures it is
polled at 1/10 cadence and an `err` record is written once per minute; it
re-arms automatically on the next success.

### 2.8 Probe records (`probe-*.jsonl`, head node)

```json
{"type":"request","t":…,"m":…,"req":"tt-20260828-153012-1","model":"deepseek-v4-flash-0731","prompt_chars":2312,
 "max_tokens":1000,"thinking":false,"stream":true,"return_routed_experts":false,"t_send":…}
{"type":"chunk","t":…,"m":…,"req":"tt-…","i":0,"bytes":181,"chars":3,"finish":null}
{"type":"response","t":…,"m":…,"req":"tt-…","t_first_chunk":…,"t_last_chunk":…,"chunks":93,
 "usage":{"prompt_tokens":512,"completion_tokens":1000,"prompt_tokens_details":{"cached_tokens":0}},
 "routed_experts_file":"probe-20260828-tt-…-experts.npy","routed_experts_shape":[1511,43,6]}
```

`chunk.chars` is the decoded text length of the delta; `bytes` is the raw
SSE line length. Per-chunk *token* counts are not carried by the OpenAI
stream; with MTP one chunk carries every token accepted in that step, so
`chars` + the final `usage.completion_tokens` + the vLLM `step_tokens`
delta between the bracketing samples give tokens-per-step.

### 2.9 Expert log (runner hotfix, each node)

```
~/.cache/huggingface/tokentrace/experts-<host>-r<rank>-<pid>-<utc>.idx.jsonl
~/.cache/huggingface/tokentrace/experts-<host>-r<rank>-<pid>-<utc>.u8
```

Index: first line `{"type":"meta","layers":43,"topk":6,"max_tokens":8192,…}`,
then one line per engine step:
`{"step":300,"t":…,"n":6,"off":504132,"len":1548,"req":["chatcmpl-…"],"sched":[6],"pos":[983],"draft":[5],"sampled":[2],"rejected":[4]}`.
`.u8` holds `n × 43 × 6` expert ids per step at byte `off`. Rows are in
request order; for request *i* the first `sampled[i]` of its `sched[i]`
rows are real tokens at positions `pos[i]…`, the rest rejected drafts.
`analyze` maps steps to a probe request by time window and reports
distinct experts per step / per request and the token-to-token switch
fraction (§3).

## 3. Derived per-step record (analysis output, not stored by the sampler)

`tokentrace analyze` merges both nodes and emits one row per engine step:

| field | meaning |
|---|---|
| `step`, `t_start`, `t_end`, `dur_ms` | from the vLLM step counter, refined by probe chunk timestamps |
| `phase` | `prefill` (no output chunk yet / prompt tokens moved), `decode`, `mixed` (running>1 with prefill), `idle` |
| `tokens_out`, `drafts`, `accepted`, `accept_ratio` | from `step_tokens` / spec-decode counter deltas |
| `ib_tx_bytes`, `ib_rx_bytes`, `ib_pkts` — per node | fabric traffic inside the step |
| `ib_busy_frac` | fraction of 20 ms bins in the step with IB traffic (continuous vs. bursty) |
| `gpu_util`, `gpu_mem_util`, `gpu_power_w` — per node | mean over the step's samples |
| `disk_rd_bytes`, `disk_wr_bytes`, `swap_in_pages`, `majfaults` — per node | should be 0 in steady decode; anything else is a wait |
| `worker_cpu_ms`, `engine_cpu_ms`, `worker_vcsw` | CPU time consumed by the worker / engine core in the step; voluntary context switches ≈ blocking waits |
| `client_gap_ms` | time between this step's chunk and the previous one, as seen by the client |
| `wait_class` | heuristic: `fabric-latency` (IB busy, GPU util low, no disk), `gpu-compute` (GPU util high), `disk/swap` (disk or swap activity), `cpu/sched` (worker blocked, nothing else moving), `client` (gap ≫ step) |

And a per-request summary: TTFT decomposition (queue → prefill steps →
first chunk), decode tok/s, bytes-per-token on the fabric, expert usage
histogram per layer (when routed experts are available), number of distinct
experts touched per step (→ the "small GPUs distributed" picture).

## 4. Hypothesis → evidence map

| Claim | Evidence in the data |
|---|---|
| Experts are all resident; no reload | `disk_rd_bytes ≈ 0`, `majfaults ≈ 0`, `swap_in ≈ 0` in every decode step while `routed_experts` show many distinct experts per step (6×43 per token, changing token to token). GPU-held memory (`gpu.procs[].mem_bytes`) constant. |
| Inter-node traffic is small | `ib_tx_bytes` per step ≈ 12.8 MB ≪ link × step (200 Gb/s × 74 ms = 1.85 GB); link utilisation < 1 % |
| Decode is latency-bound on the fabric | `ib_busy_frac` ≈ 1 with tiny packets (≈ 51 B/packet ⇒ 248 k pkt/s), `gpu_util` well below 100 %, `worker_vcsw` high; step time scales with #collectives, not bytes |
| Efficiency comes from MTP | `accept_ratio` ≈ 0.69, `tokens_out/step` ≈ 2–3.5, so 13.5 steps/s ⇒ 30–50 tok/s |
| Load cost is pre-paid | during `model_load_*`: `disk_rd_bytes` ≈ weights size, `gpu.procs[].mem_bytes` ramp, `cpu` busy (conversion), followed by zero disk traffic in serving |

## 5. Model load visualisation (later)

The sampler is process-external, so it needs nothing new for a load: run it
before `./start-deepseek-v4-flash-dspark.sh`, drop `mark model_load_start`,
and `mark model_ready` when `/health` turns 200. The slow records give the
NVMe read ramp (weights), the meminfo/NVML memory ramp (conversion to the
GPU-resident format), CPU and fabric activity (NCCL init, CUDA-graph
capture shows as GPU util bursts with zero fabric traffic). Same schema, no
new record types.

## 6. Hardening rules (before it touches the live cluster)

1. Read-only sources only; the sampler never writes anywhere but its own
   trace directory. No `docker exec`, no nvidia-smi subprocesses (NVML via
   ctypes), no ssh.
2. Bounded resources: one thread per cadence, fixed-size buffers, fast
   cadence only while armed (activity or probe), file rotation per day,
   refuse to start when the trace filesystem has < 2 GB free, and stop
   writing (but keep running) when it drops below 500 MB.
3. Every source is isolated by try/except with an error budget; a
   permanently missing source degrades to `err` records, never a crash.
4. Clean shutdown on SIGTERM/SIGINT: flush + fsync; a partial last line is
   tolerated by the reader.
5. Unit tests run on macOS against fixture sysfs/proc trees and a fake NVML;
   the only Linux-only test is the smoke run on the nodes themselves.
6. Soak test on both nodes for ≥ 10 min with the server idle, measuring
   the sampler's own CPU (target < 5 % of one core) and checking that the
   vLLM decode rate with the sampler running equals the rate without it.
7. The expert recorder is gated behind `DSPARK_TOKENTRACE_EXPERTS=1` in
   compose (hotfix applied at boot only when set, byte-identical boot
   otherwise), self-tested on the real GPU in a throwaway container
   (`scripts/test-tokentrace-experts-gpu.py`: in-graph capture, replay,
   D2H, ring reuse, clamping, exception path) before the one restart that
   enables it. First live run: 2026-08-28, see
   `results/TOKENTRACE-E2E-2026-08-28.md`.

## 7. Video (`python3 -m tokentrace.video`)

One frame per 1/30 s of slowed-down generation (default ×0.25), rendered
with Pillow and encoded by ffmpeg. Inputs: the `analyze --out` JSON, the
per-chunk text/token file (`<req>-chunks.json`, from vLLM `/tokenize`) and
the runner expert log. What it shows and why:

| Element | Encoding | Reason |
|---|---|---|
| 43 × 256 matrix | one cell per (layer, routed expert) | both nodes compute every routed expert (half of its width each, TP=2), and nothing distinguishes which half fed the accepted token — so the map is one map, identical on both nodes; node-level state (paging) lives on the edge strips |
| cell brightness | +0.25 per use, saturates at 1, decays with τ = 2.5 s of generation time; never-used stays dark | "used → bright and saturates, unused → fades, never used → dark" |
| white flash | experts of the current step's **accepted** tokens | the work that produced text |
| amber flash | experts touched only by **rejected** drafts this step | the wasted verification rows |
| red edge strip | left = dgx01, right = dgx02, red while that node pages (swap-in / major fault) | the only asymmetric stall source in this topology |
| step class | FAST (≥5 accepted, ≤1.1× baseline) green · NORMAL grey-blue · STALL (>1.15× baseline ≈ slowest 10 %) orange | separates "as fast as the pipeline goes" from time excursions; paging and rejects are tints, not classes, because they cost ~1 ms |
| gauges | last 120 step bars + baseline; current class, ms, accepted/rows, tok/s; RoCE MB / packets / Gb/s; GPU W; paging; experts this step and cumulative coverage | the per-step evidence behind the class |
| text stream | 4 scrolling lines, coloured by the class of the step that produced each chunk; for the newest token, 43 bars = how many of that layer's 6 experts the previous token also used (routing continuity, 36 % observed vs 2 % random; layers 0–2 ≈ 0, layers 24–26 ≈ 3/6) plus the step's distinct pairs vs slots | weights are read per step, not per token, so the within-step overlap (41 %) is what sets the 6 GB/step; the per-token continuity is the model property behind it |

No expert ever crosses a node or disk boundary in this deployment, so
"boundary crossing" is not a stall class; the stalls that exist are
speculation rejects, step-time excursions and background paging.
