#!/usr/bin/env python3
"""Hotfix (tokentrace): per-step expert-routing log from the V2 GPU model
runner. See docs/TOKENTRACE.md and patches/tokentrace_experts_runtime.py.

What it does
------------
1. Installs ``patches/tokentrace_experts_runtime.py`` as
   ``vllm/v1/worker/gpu/tokentrace_experts.py`` (verbatim copy).
2. Patches ``vllm/v1/worker/gpu/model_runner.py``:
   - ``initialize_kv_cache``: after the KV connector is created (model
     loaded, CUDA graphs not yet captured) → ``self.tokentrace =
     maybe_init_expert_trace(self)``, which binds a capture callback to every
     MoE router (``BaseRouter.set_capture_fn``) so the graph capture that
     follows includes the copy of ``topk_ids`` into a fixed device buffer.
   - ``sample_tokens``: right after ``self.sample(...)`` → one non-blocking
     D2H of the routing buffer + accepted token ids + sampler counts, drained
     to disk when the CUDA event says it landed.
   - ``shutdown``: flush + close.

Runtime gate: the code path is a no-op unless the container env has
``DSPARK_TOKENTRACE_EXPERTS=1`` (the patch itself is only applied when the
compose entrypoint sees that flag, so a default boot is byte-identical).

Idempotent; anchors are asserted; ``--status`` reports.
"""
from pathlib import Path
import shutil
import sys

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
RUNNER = VLLM / "v1/worker/gpu/model_runner.py"
TARGET = VLLM / "v1/worker/gpu/tokentrace_experts.py"
SRC_CANDIDATES = [Path("/opt/dspark-patches/tokentrace_experts_runtime.py"),
                  Path(__file__).resolve().parent / "tokentrace_experts_runtime.py"]
MARK = "# [tokentrace-hotfix]"

if len(sys.argv) > 1 and sys.argv[1] == "--status":
    src = RUNNER.read_text() if RUNNER.is_file() else ""
    print("tokentrace expert recorder (V2 runner):",
          "APPLIED" if MARK in src and TARGET.is_file() else "NOT APPLIED")
    raise SystemExit(0)

src_file = next((p for p in SRC_CANDIDATES if p.is_file()), None)
assert src_file is not None, "tokentrace: runtime module tokentrace_experts_runtime.py not found"
shutil.copyfile(src_file, TARGET)
print(f"[tokentrace-hotfix] installed {TARGET}")

src = RUNNER.read_text()
if MARK in src:
    print(f"[tokentrace-hotfix] already applied to {RUNNER}")
    raise SystemExit(0)

# 1. import
A_IMPORT = "import functools\nimport gc\nimport time\n"
assert A_IMPORT in src, "tokentrace: import anchor not found; refusing to patch"
src = src.replace(A_IMPORT, A_IMPORT + f"from vllm.v1.worker.gpu.tokentrace_experts import maybe_init_expert_trace  {MARK}\n", 1)

# 2. init after KV connector (model loaded, before capture_model)
A_INIT = "        self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)\n"
assert src.count(A_INIT) == 1, "tokentrace: initialize_kv_cache anchor not found"
src = src.replace(A_INIT, A_INIT + f"        self.tokentrace = maybe_init_expert_trace(self)  {MARK}\n", 1)

# 3. record after sampling
A_SAMPLE = ("        sampler_output, num_sampled, num_rejected = self.sample(\n"
            "            hidden_states, input_batch, grammar_output\n"
            "        )\n")
assert src.count(A_SAMPLE) == 1, "tokentrace: sample_tokens anchor not found"
src = src.replace(A_SAMPLE, A_SAMPLE +
                  f"        if getattr(self, \"tokentrace\", None) is not None:  {MARK}\n"
                  f"            self.tokentrace.record(input_batch, sampler_output.sampled_token_ids, num_sampled, num_rejected)  {MARK}\n", 1)

# 4. shutdown flush
A_SHUT = ("    def shutdown(self) -> None:\n"
          "        \"\"\"Release GPU tensors (model weights, KV caches, workspace) so that\n"
          "        memory is reclaimable when running in the same process.\"\"\"\n")
assert src.count(A_SHUT) == 1, "tokentrace: shutdown anchor not found"
src = src.replace(A_SHUT, A_SHUT +
                  f"        if getattr(self, \"tokentrace\", None) is not None:  {MARK}\n"
                  f"            self.tokentrace.close()  {MARK}\n"
                  f"            self.tokentrace = None  {MARK}\n", 1)

RUNNER.write_text(src)
import py_compile
py_compile.compile(str(RUNNER), doraise=True)
py_compile.compile(str(TARGET), doraise=True)
print(f"[tokentrace-hotfix] applied to {RUNNER}")
