"""tokentrace.nvml — the ctypes binding against a fake libnvidia-ml that
behaves like GB10 (util/power/clock/procs/samples OK, memory info
unsupported) (pytest)."""
import ctypes
import time

import pytest

from tokentrace import nvml as nv


class FakeLib:
    """Implements the NVML entry points tokentrace.nvml calls. ``byref``
    arguments expose the underlying object as ``._obj``."""

    def __init__(self, gb10=True):
        self.gb10 = gb10
        self.calls = []
        self.shutdown_called = False
        self.samples = {nv.SAMPLE_GPU_UTIL: [(1_787_852_246_100_000 + i * 200_000, 10 * i) for i in range(5)],
                        nv.SAMPLE_MEM_UTIL: [(1_787_852_246_100_000, 0)]}

    def nvmlInit_v2(self):
        return 0

    def nvmlShutdown(self):
        self.shutdown_called = True
        return 0

    def nvmlDeviceGetHandleByIndex_v2(self, idx, h):
        h._obj.value = 42
        return 0

    def nvmlDeviceGetName(self, h, buf, n):
        buf.value = b"NVIDIA GB10"
        return 0

    def nvmlSystemGetNVMLVersion(self, buf, n):
        buf.value = b"580.173"
        return 0

    def nvmlDeviceGetUtilizationRates(self, h, u):
        u._obj.gpu, u._obj.memory = 95, 0
        return 0

    def nvmlDeviceGetPowerUsage(self, h, p):
        p._obj.value = 24500
        return 0

    def nvmlDeviceGetClockInfo(self, h, kind, c):
        if kind == nv.NVML_CLOCK_SM:
            c._obj.value = 2177
            return 0
        return nv.NVML_ERROR_NOT_SUPPORTED

    def nvmlDeviceGetComputeRunningProcesses_v3(self, h, cnt, arr):
        if arr is None:
            cnt._obj.value = 1
            return 7  # INSUFFICIENT_SIZE, as the real driver does when cnt was 0
        arr[0].pid, arr[0].usedGpuMemory = 306134, 107933581312
        cnt._obj.value = 1
        return 0

    def nvmlDeviceGetSamples(self, h, kind, since, vt, cnt, arr):
        self.calls.append(("samples", kind, since.value))
        data = [s for s in self.samples.get(kind, []) if s[0] > since.value]
        if kind not in self.samples:
            return nv.NVML_ERROR_NOT_SUPPORTED
        if not data:
            return nv.NVML_ERROR_NOT_FOUND
        vt._obj.value = 1  # unsigned int
        cnt._obj.value = len(data)
        if arr is not None:
            for i, (ts, v) in enumerate(data):
                arr[i].timeStamp = ts
                arr[i].sampleValue.uiVal = v
        return 0


@pytest.fixture
def fake(monkeypatch):
    lib = FakeLib()
    monkeypatch.setattr(ctypes, "CDLL", lambda name: lib)
    return lib


def test_missing_library_is_not_fatal():
    n = nv.Nvml(libname="/definitely/not/libnvidia-ml.so")
    assert n.available is False and n.util() is None if n.lib else True
    n.shutdown()  # no-op


def test_capabilities_probe_on_gb10(fake):
    n = nv.Nvml()
    assert n.available and n.name == "NVIDIA GB10" and n.version == "580.173"
    assert n.supports["util"] and n.supports["power"] and n.supports["sm_clock"] and n.supports["procs"]
    assert n.supports["samples"] == ["gpu_util", "mem_util"]  # power / sm_clk unsupported
    assert n.util() == (95, 0) and n.power_mw() == 24500 and n.sm_mhz() == 2177
    assert n.compute_procs() == [{"pid": 306134, "mem_bytes": 107933581312}]
    n.shutdown()
    assert fake.shutdown_called


def test_samples_are_incremental(fake):
    n = nv.Nvml()
    n._last_sample_ts.clear()
    s = n.samples("gpu_util", 0)
    assert [v for _, v in s] == [0, 10, 20, 30, 40]
    assert n._last_sample_ts["gpu_util"] == s[-1][0]
    assert n.samples("gpu_util") == []  # nothing newer than the last drain → NOT_FOUND → []
    fake.samples[nv.SAMPLE_GPU_UTIL].append((s[-1][0] + 200_000, 50))
    assert n.samples("gpu_util") == [(s[-1][0] + 200_000, 50)]
    assert n.samples("power") is None  # unsupported kind


def test_sample_timestamp_conversion():
    now = time.time()
    assert nv.Nvml.sample_ts_to_wall(int(now * 1e6) - 250_000, now) == pytest.approx(now - 0.25, abs=1e-3)
    assert nv.Nvml.sample_ts_to_wall(12345, now) == now  # non-epoch base → fall back to now


def test_init_failure_paths(monkeypatch):
    class Bad(FakeLib):
        def nvmlInit_v2(self):
            return 999
    monkeypatch.setattr(ctypes, "CDLL", lambda name: Bad())
    assert nv.Nvml().available is False

    class NoHandle(FakeLib):
        def nvmlDeviceGetHandleByIndex_v2(self, idx, h):
            return 6
    monkeypatch.setattr(ctypes, "CDLL", lambda name: NoHandle())
    assert nv.Nvml().available is False
