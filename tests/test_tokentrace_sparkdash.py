import json
from datetime import timezone

import pytest

from tokentrace.sparkdash import resolve_pair
from tokentrace.video import format_timecode


def registry():
    return {
        "sparks": [
            {
                "id": "cluster-head",
                "role": "head",
                "lanIp": "10.0.0.11",
                "ssh": {"host": "dgx01", "user": "ubuntu", "auth": "key"},
            },
            {
                "id": "cluster-worker",
                "role": "worker",
                "workerHeadId": "cluster-head",
                "lanIp": "10.0.0.12",
                "cx7Ip": "192.0.2.40",
                "ssh": {"host": "dgx02", "user": "ubuntu", "auth": "key"},
            },
        ]
    }


def test_resolve_pair_uses_roles_links_and_sparkdash_addresses():
    assert resolve_pair(registry()) == {
        "head_id": "cluster-head",
        "worker_id": "cluster-worker",
        "head_ssh": "ubuntu@dgx01",
        "worker_ssh": "ubuntu@dgx02",
        "worker_peer": "192.0.2.40",
    }


def test_resolve_pair_rejects_ambiguous_workers():
    data = registry()
    data["sparks"].append({
        "id": "other-worker", "role": "worker", "workerHeadId": "cluster-head",
        "lanIp": "10.0.0.13", "ssh": {"host": "dgx03"},
    })
    with pytest.raises(ValueError, match="exactly one worker"):
        resolve_pair(data)


def test_resolve_pair_infers_legacy_head_from_worker_link():
    data = registry()
    data["sparks"][0]["role"] = "standalone"
    assert resolve_pair(data)["head_id"] == "cluster-head"


def test_timecode_is_wall_clock_with_centiseconds():
    assert format_timecode(23 * 3600 + 18 * 60 + 25.289, timezone.utc) == "23:18:25.28"
