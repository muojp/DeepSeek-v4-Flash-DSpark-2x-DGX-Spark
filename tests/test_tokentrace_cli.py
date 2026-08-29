"""python -m tokentrace — sub-command dispatch, mark/stat against a control
server (pytest)."""
import json

from conftest import free_port
from tokentrace import __main__ as cli
from tokentrace import control


def test_usage_and_unknown_command(capsys):
    assert cli.main([]) == 2
    assert cli.main(["--help"]) == 2
    assert cli.main(["nonsense"]) == 2
    assert "usage" in capsys.readouterr().err


def test_targets_parsing():
    assert cli._targets([]) == [("127.0.0.1", control.DEFAULT_PORT)]
    assert cli._targets(["10.0.0.2", "10.0.0.3:5"]) == [("10.0.0.2", control.DEFAULT_PORT), ("10.0.0.3", 5)]
    assert cli._targets([":9"]) == [("127.0.0.1", 9)]


def test_mark_and_stat_roundtrip(capsys):
    marks = []
    srv = control.ControlServer("127.0.0.1", 0, on_arm=lambda *_: None, on_mark=lambda l, n, who: marks.append((l, n)),
                                on_stat=lambda: {"host": "h", "armed": False})
    port = srv.sock.getsockname()[1]
    srv.start()
    try:
        assert cli.main(["mark", "model_load_start", "compose", "up", "--to", f"127.0.0.1:{port}"]) == 0
        assert marks == [("model_load_start", "compose up")]
        assert cli.main(["stat", f"127.0.0.1:{port}"]) == 0
        out = capsys.readouterr().out.strip().splitlines()
        assert json.loads(out[-1])["reply"]["host"] == "h"
        assert cli.main(["mark"]) == 2
    finally:
        srv.stop()
    dead = free_port()
    assert cli.main(["mark", "x", "--to", f"127.0.0.1:{dead}"]) == 1  # unreachable target → nonzero
    assert cli.main(["stat", f"127.0.0.1:{dead}"]) == 0 and json.loads(capsys.readouterr().out.strip().splitlines()[-1])["reply"] is None


def test_subcommands_dispatch_to_modules(monkeypatch):
    seen = {}
    for name, mod in (("sampler", "tokentrace.sampler"), ("probe", "tokentrace.probe"), ("analyze", "tokentrace.analyze")):
        import importlib
        m = importlib.import_module(mod)
        monkeypatch.setattr(m, "main", lambda rest, _n=name: (seen.__setitem__(_n, rest), 0)[1])
        assert cli.main([name, "--x", "1"]) == 0
    assert seen == {"sampler": ["--x", "1"], "probe": ["--x", "1"], "analyze": ["--x", "1"]}
