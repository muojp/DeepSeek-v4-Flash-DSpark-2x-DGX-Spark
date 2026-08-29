"""``python3 -m tokentrace <sampler|probe|analyze|mark|stat> [args]``"""
from __future__ import annotations

import json
import sys

from . import control

USAGE = "usage: python3 -m tokentrace {sampler|probe|analyze|mark|stat} [args]\n"


def _targets(items: list[str]) -> list[tuple[str, int]]:
    out = []
    for it in items:
        h, _, p = it.partition(":")
        out.append((h or "127.0.0.1", int(p or control.DEFAULT_PORT)))
    return out or [("127.0.0.1", control.DEFAULT_PORT)]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        sys.stderr.write(USAGE)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "sampler":
        from .sampler import main as m
        return m(rest)
    if cmd == "probe":
        from .probe import main as m
        return m(rest)
    if cmd == "analyze":
        from .analyze import main as m
        return m(rest)
    if cmd == "mark":
        # mark <label> [note] [--to host[:port],host[:port]]
        to = []
        if "--to" in rest:
            i = rest.index("--to")
            to = rest[i + 1].split(",") if i + 1 < len(rest) else []
            rest = rest[:i] + rest[i + 2:]
        if not rest:
            sys.stderr.write("usage: mark <label> [note] [--to host[:port],...]\n")
            return 2
        res = control.mark(_targets(to), rest[0], " ".join(rest[1:]))
        print(json.dumps(res))
        return 0 if all(res.values()) else 1
    if cmd == "stat":
        to = rest[0].split(",") if rest else []
        for h, p in _targets(to):
            print(json.dumps({"target": f"{h}:{p}", "reply": control.send(h, p, {"op": "stat"})}))
        return 0
    sys.stderr.write(USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main())
