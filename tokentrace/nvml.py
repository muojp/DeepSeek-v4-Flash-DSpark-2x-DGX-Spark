"""Minimal NVML binding via ctypes (no pynvml on the hosts).

GB10 facts (driver 580.173): utilisation rates, power, SM clock and the
compute-process list work; ``GetMemoryInfo`` / PCIe throughput return
NOT_SUPPORTED (unified memory); the util / mem-util sample ring ticks every
200 ms. Everything here degrades to ``None`` instead of raising once
``init()`` has succeeded; ``Nvml.available`` is False when the library is
missing (macOS / CI).
"""
from __future__ import annotations

import ctypes
import time

NVML_SUCCESS = 0
NVML_ERROR_NOT_SUPPORTED = 3
NVML_ERROR_NOT_FOUND = 6

# nvmlSamplingType_t
SAMPLE_TOTAL_POWER = 0
SAMPLE_GPU_UTIL = 1
SAMPLE_MEM_UTIL = 2
SAMPLE_PROC_CLK = 5
SAMPLE_MEM_CLK = 6
SAMPLE_KINDS = {"gpu_util": SAMPLE_GPU_UTIL, "mem_util": SAMPLE_MEM_UTIL,
                "power": SAMPLE_TOTAL_POWER, "sm_clk": SAMPLE_PROC_CLK, "mem_clk": SAMPLE_MEM_CLK}

NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2


class _Util(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _Value(ctypes.Union):
    _fields_ = [("dVal", ctypes.c_double), ("uiVal", ctypes.c_uint), ("ulVal", ctypes.c_ulong),
                ("ullVal", ctypes.c_ulonglong), ("sllVal", ctypes.c_longlong)]


class _Sample(ctypes.Structure):
    _fields_ = [("timeStamp", ctypes.c_ulonglong), ("sampleValue", _Value)]


class _ProcInfo(ctypes.Structure):  # nvmlProcessInfo_v2/v3 layout
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


class Nvml:
    def __init__(self, libname: str = "libnvidia-ml.so.1", index: int = 0):
        self.available = False
        self.lib = None
        self.handle = ctypes.c_void_p()
        self.supports: dict[str, bool] = {}
        self.name = None
        self.version = None
        self._last_sample_ts: dict[str, int] = {}
        self._errors: dict[str, int] = {}
        try:
            self.lib = ctypes.CDLL(libname)
        except OSError:
            return
        if self.lib.nvmlInit_v2() != NVML_SUCCESS:
            self.lib = None
            return
        if self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle)) != NVML_SUCCESS:
            self.lib = None
            return
        self.available = True
        buf = ctypes.create_string_buffer(96)
        if self.lib.nvmlDeviceGetName(self.handle, buf, 96) == NVML_SUCCESS:
            self.name = buf.value.decode(errors="replace")
        vb = ctypes.create_string_buffer(80)
        if self.lib.nvmlSystemGetNVMLVersion(vb, 80) == NVML_SUCCESS:
            self.version = vb.value.decode(errors="replace")
        self._probe_support()

    # ── capability probe ─────────────────────────────────────────────
    def _probe_support(self):
        self.supports["util"] = self.util() is not None
        self.supports["power"] = self.power_mw() is not None
        self.supports["sm_clock"] = self.sm_mhz() is not None
        self.supports["procs"] = self.compute_procs() is not None
        ok = []
        for k in ("gpu_util", "mem_util", "power", "sm_clk"):
            if self.samples(k, 0) is not None:
                ok.append(k)
        self.supports["samples"] = ok

    # ── instantaneous ────────────────────────────────────────────────
    def util(self) -> tuple[int, int] | None:
        u = _Util()
        if self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(u)) != NVML_SUCCESS:
            return None
        return int(u.gpu), int(u.memory)

    def power_mw(self) -> int | None:
        p = ctypes.c_uint()
        if self.lib.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(p)) != NVML_SUCCESS:
            return None
        return int(p.value)

    def sm_mhz(self) -> int | None:
        c = ctypes.c_uint()
        if self.lib.nvmlDeviceGetClockInfo(self.handle, NVML_CLOCK_SM, ctypes.byref(c)) != NVML_SUCCESS:
            return None
        return int(c.value)

    def compute_procs(self) -> list[dict] | None:
        cnt = ctypes.c_uint(0)
        r = self.lib.nvmlDeviceGetComputeRunningProcesses_v3(self.handle, ctypes.byref(cnt), None)
        if r not in (NVML_SUCCESS, 7):  # 7 = INSUFFICIENT_SIZE (expected when cnt>0)
            return None
        n = max(int(cnt.value), 1)
        arr = (_ProcInfo * (n + 4))()
        cnt = ctypes.c_uint(n + 4)
        r = self.lib.nvmlDeviceGetComputeRunningProcesses_v3(self.handle, ctypes.byref(cnt), arr)
        if r != NVML_SUCCESS:
            return None
        return [{"pid": int(arr[i].pid), "mem_bytes": int(arr[i].usedGpuMemory)} for i in range(int(cnt.value))]

    # ── ring-buffer samples ──────────────────────────────────────────
    def samples(self, kind: str, since_us: int | None = None) -> list[tuple[int, int]] | None:
        """[(timestamp_us, value)] newer than ``since_us`` (default: last
        drain for this kind). Timestamps are NVML's CPU microsecond stamps."""
        st = SAMPLE_KINDS[kind]
        if since_us is None:
            since_us = self._last_sample_ts.get(kind, 0)
        vt = ctypes.c_uint()
        cnt = ctypes.c_uint(0)
        r = self.lib.nvmlDeviceGetSamples(self.handle, st, ctypes.c_ulonglong(since_us),
                                          ctypes.byref(vt), ctypes.byref(cnt), None)
        if r == NVML_ERROR_NOT_FOUND:
            return []
        if r != NVML_SUCCESS or cnt.value == 0:
            return None if r != NVML_SUCCESS else []
        n = int(cnt.value)
        arr = (_Sample * n)()
        r = self.lib.nvmlDeviceGetSamples(self.handle, st, ctypes.c_ulonglong(since_us),
                                          ctypes.byref(vt), ctypes.byref(cnt), arr)
        if r == NVML_ERROR_NOT_FOUND:
            return []
        if r != NVML_SUCCESS:
            return None
        out = []
        for i in range(int(cnt.value)):
            s = arr[i]
            if vt.value == 0:
                v = s.sampleValue.dVal
            elif vt.value == 1:
                v = s.sampleValue.uiVal
            elif vt.value == 2:
                v = s.sampleValue.ulVal
            elif vt.value == 3:
                v = s.sampleValue.ullVal
            else:
                v = s.sampleValue.sllVal
            out.append((int(s.timeStamp), v))
        if out:
            self._last_sample_ts[kind] = out[-1][0]
        return out

    @staticmethod
    def sample_ts_to_wall(ts_us: int, now_wall: float | None = None) -> float:
        """NVML sample stamps are CPU microseconds since the epoch on Linux.
        Guard against a non-epoch base by falling back to 'now'."""
        now_wall = time.time() if now_wall is None else now_wall
        t = ts_us / 1e6
        return t if abs(t - now_wall) < 86400 else now_wall

    def shutdown(self):
        if self.lib is not None:
            try:
                self.lib.nvmlShutdown()
            except Exception:
                pass
