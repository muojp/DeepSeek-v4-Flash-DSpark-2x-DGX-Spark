"""tokentrace — process-external per-token telemetry for the DSpark stack.

See docs/TOKENTRACE.md for the record schema and the hypothesis map.
Stdlib only; runs on the DGX Spark hosts' python3 (3.12) with no extra
packages. Sub-commands: ``sampler``, ``probe``, ``analyze``, ``mark``.
"""

SCHEMA_VERSION = 1
