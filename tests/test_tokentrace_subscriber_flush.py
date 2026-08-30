import fcntl
import importlib.util
import os
import sys
import types
from pathlib import Path


def load_runtime(monkeypatch):
    torch = types.ModuleType("torch")
    vllm = types.ModuleType("vllm")
    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = lambda _name: types.SimpleNamespace(
        info=lambda *_a, **_k: None,
        warning=lambda *_a, **_k: None,
        debug=lambda *_a, **_k: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)
    path = Path(__file__).parents[1] / "patches" / "tokentrace_experts_runtime.py"
    spec = importlib.util.spec_from_file_location("tokentrace_experts_lock_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_only_subscriber_lock_selects_fast_flush(tmp_path, monkeypatch):
    module = load_runtime(monkeypatch)
    idx_path = tmp_path / "experts-test.idx.jsonl"
    idx_path.write_text("{}\n")
    recorder = module.ExpertTraceRecorder.__new__(module.ExpertTraceRecorder)
    recorder.idx = open(idx_path, "a", encoding="utf-8")

    assert recorder._subscriber_active() is False
    reader = open(idx_path, "r", encoding="utf-8")
    fcntl.flock(reader.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
    try:
        assert recorder._subscriber_active() is True
    finally:
        fcntl.flock(reader.fileno(), fcntl.LOCK_UN)
        reader.close()
        recorder.idx.close()


def test_flush_intervals_are_positive_and_env_overridable(monkeypatch):
    module = load_runtime(monkeypatch)
    monkeypatch.setenv("DSPARK_TOKENTRACE_FLUSH_S", "3.5")
    monkeypatch.setenv("DSPARK_TOKENTRACE_SUBSCRIBER_FLUSH_S", "0.025")
    assert module._positive_float_env("DSPARK_TOKENTRACE_FLUSH_S", 2.0) == 3.5
    assert module._positive_float_env("DSPARK_TOKENTRACE_SUBSCRIBER_FLUSH_S", 0.05) == 0.025
    monkeypatch.setenv("DSPARK_TOKENTRACE_FLUSH_S", "0")
    assert module._positive_float_env("DSPARK_TOKENTRACE_FLUSH_S", 2.0) == 2.0
