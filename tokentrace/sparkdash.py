"""Resolve a tokentrace head/worker pair from sparkDash's unit registry."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def _role(unit: dict) -> str:
    role = unit.get("role")
    if role in ("head", "worker", "standalone"):
        return role
    return "worker" if unit.get("workerNode") else "standalone"


def _ssh_host(unit: dict) -> str:
    ssh = unit.get("ssh") or {}
    return str(ssh.get("host") or unit.get("lanIp") or "").strip()


def _ssh_target(unit: dict) -> str:
    host = _ssh_host(unit)
    user = str((unit.get("ssh") or {}).get("user") or "").strip()
    return f"{user}@{host}" if user else host


def resolve_pair(payload: dict, head_id: str | None = None) -> dict:
    """Return the linked head/worker pair and their SSH/fabric addresses.

    sparkDash is the source of truth for machine identity.  The worker must
    either name its head through ``workerHeadId`` or be the only worker when
    there is exactly one head.
    """
    units = [u for u in payload.get("sparks", []) if isinstance(u, dict)]
    workers_all = [u for u in units if _role(u) == "worker"]
    explicit_heads = [u for u in units if _role(u) == "head"]
    referenced_ids = {u.get("workerHeadId") for u in workers_all if u.get("workerHeadId")}
    inferred_heads = [u for u in units if u.get("id") in referenced_ids]
    heads = explicit_heads or inferred_heads
    if head_id:
        heads = [u for u in heads if u.get("id") == head_id]
    if len(heads) != 1:
        detail = f" for id {head_id!r}" if head_id else ""
        raise ValueError(f"sparkDash must contain exactly one head{detail}; found {len(heads)}")
    head = heads[0]
    workers = [u for u in workers_all if u.get("workerHeadId") in (None, "", head.get("id"))]
    linked = [u for u in workers if u.get("workerHeadId") == head.get("id")]
    if linked:
        workers = linked
    if len(workers) != 1:
        raise ValueError(
            f"sparkDash head {head.get('id')!r} must have exactly one worker; found {len(workers)}"
        )
    worker = workers[0]
    head_ssh, worker_ssh = _ssh_host(head), _ssh_host(worker)
    if not head_ssh or not worker_ssh:
        raise ValueError("sparkDash head and worker both need ssh.host or lanIp")
    peer = str(worker.get("cx7Ip") or worker_ssh).strip()
    return {
        "head_id": str(head.get("id")),
        "worker_id": str(worker.get("id")),
        "head_ssh": _ssh_target(head),
        "worker_ssh": _ssh_target(worker),
        "worker_peer": peer,
    }


def fetch_registry(url: str, timeout: float = 5.0) -> dict:
    endpoint = url.rstrip("/")
    if not endpoint.endswith("/api/sparks"):
        endpoint += "/api/sparks"
    with urllib.request.urlopen(endpoint, timeout=timeout) as response:
        return json.load(response)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python3 -m tokentrace.sparkdash")
    p.add_argument("--url", required=True, help="sparkDash base URL or /api/sparks URL")
    p.add_argument("--head-id", default=None)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--shell", action="store_true", help="emit one tab-separated row for deploy.sh")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        pair = resolve_pair(fetch_registry(a.url, a.timeout), a.head_id)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"tokentrace: cannot resolve Spark pair from {a.url}: {exc}", file=sys.stderr)
        return 1
    if a.shell:
        print("\t".join(pair[k] for k in ("head_ssh", "worker_ssh", "worker_peer", "head_id", "worker_id")))
    else:
        print(json.dumps(pair, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
