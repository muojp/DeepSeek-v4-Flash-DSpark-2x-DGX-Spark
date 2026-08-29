"""UDP control channel shared by the samplers and the probe.

One datagram = one JSON object. Ops:
  {"op":"ping","t0":<sender wall>}   → {"op":"pong","t0":..,"t2":<recv wall>,"t3":<send wall>}
  {"op":"arm","secs":30}             → keep the fast cadence on for ``secs``
  {"op":"mark","label":"..","note":".."} → write a mark record
  {"op":"stat"}                       → {"op":"stat", ...writer/sampler stats}

Clock offset uses the NTP four-timestamp formula over the RoCE fabric
(RTT ≈ 0.1 ms), which is what makes cross-node step alignment possible.
"""
from __future__ import annotations

import json
import socket
import statistics
import threading
import time

DEFAULT_PORT = 47001
MAX_DGRAM = 4096


def ntp_offset(t0: float, t1: float, t2: float, t3: float) -> tuple[float, float]:
    """(offset, rtt): offset = remote_clock - local_clock, seconds.
    t0 client send, t1 server recv, t2 server send, t3 client recv."""
    offset = ((t1 - t0) + (t2 - t3)) / 2.0
    rtt = (t3 - t0) - (t2 - t1)
    return offset, rtt


def summarize_offsets(pairs: list[tuple[float, float]]) -> dict | None:
    """Median offset over the lowest-RTT half of the exchanges."""
    if not pairs:
        return None
    pairs = sorted(pairs, key=lambda p: p[1])
    keep = pairs[: max(1, len(pairs) // 2)]
    return {"offset_us": int(round(statistics.median(p[0] for p in keep) * 1e6)),
            "rtt_us": int(round(min(p[1] for p in pairs) * 1e6)),
            "n": len(pairs)}


class ControlServer(threading.Thread):
    def __init__(self, bind: str, port: int, on_arm, on_mark, on_stat, clock=time.time):
        super().__init__(name="tt-control", daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind, port))
        self.sock.settimeout(0.5)
        self.on_arm, self.on_mark, self.on_stat = on_arm, on_mark, on_stat
        self.clock = clock
        self.stop_event = threading.Event()
        self.handled = 0

    def run(self):
        while not self.stop_event.is_set():
            try:
                data, addr = self.sock.recvfrom(MAX_DGRAM)
            except socket.timeout:
                continue
            except OSError:
                break
            t1 = self.clock()
            try:
                msg = json.loads(data.decode("utf-8"))
            except ValueError:
                continue
            self.handled += 1
            op = msg.get("op")
            reply = None
            try:
                if op == "ping":
                    reply = {"op": "pong", "t0": msg.get("t0"), "t1": t1, "t2": self.clock()}
                elif op == "arm":
                    self.on_arm(float(msg.get("secs", 30)), addr[0])
                    reply = {"op": "ok"}
                elif op == "mark":
                    self.on_mark(str(msg.get("label", ""))[:200], str(msg.get("note", ""))[:2000], addr[0])
                    reply = {"op": "ok"}
                elif op == "stat":
                    reply = {"op": "stat", **self.on_stat()}
            except Exception as e:  # never let a bad datagram kill the server
                reply = {"op": "err", "error": str(e)[:200]}
            if reply is not None:
                try:
                    self.sock.sendto(json.dumps(reply).encode("utf-8"), addr)
                except OSError:
                    pass

    def stop(self):
        self.stop_event.set()
        try:
            self.sock.close()
        except OSError:
            pass


def send(host: str, port: int, msg: dict, timeout: float = 0.5, clock=time.time) -> dict | None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        s.sendto(json.dumps(msg).encode("utf-8"), (host, port))
        data, _ = s.recvfrom(MAX_DGRAM)
        return json.loads(data.decode("utf-8"))
    except (OSError, ValueError):
        return None
    finally:
        s.close()


def measure_offset(host: str, port: int, n: int = 8, timeout: float = 0.3, clock=time.time) -> dict | None:
    pairs = []
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        for _ in range(n):
            t0 = clock()
            try:
                s.sendto(json.dumps({"op": "ping", "t0": t0}).encode("utf-8"), (host, port))
                data, _ = s.recvfrom(MAX_DGRAM)
            except OSError:
                continue
            t3 = clock()
            try:
                r = json.loads(data.decode("utf-8"))
            except ValueError:
                continue
            if r.get("op") != "pong" or r.get("t0") != t0:
                continue
            pairs.append(ntp_offset(t0, float(r["t1"]), float(r["t2"]), t3))
    finally:
        s.close()
    return summarize_offsets(pairs)


def arm(hosts: list[tuple[str, int]], secs: float) -> dict[str, bool]:
    return {f"{h}:{p}": send(h, p, {"op": "arm", "secs": secs}) is not None for h, p in hosts}


def mark(hosts: list[tuple[str, int]], label: str, note: str = "") -> dict[str, bool]:
    return {f"{h}:{p}": send(h, p, {"op": "mark", "label": label, "note": note}) is not None for h, p in hosts}
