"""tokentrace.sources — pure parsers and sysfs/procfs readers (pytest)."""
from pathlib import Path

from conftest import CPUSTAT, DISKSTATS, MEMINFO, PIDSTAT, PIDSTATUS, VMSTAT, vllm_metrics_text
from tokentrace import sources


def test_meminfo():
    m = sources.parse_meminfo(MEMINFO)
    assert m["MemTotal"] == 127600792
    assert m["SwapCached"] == 1273248
    assert "Hugepagesize" not in m


def test_vmstat():
    assert sources.parse_vmstat(VMSTAT) == {"pswpin": 3011, "pswpout": 7042, "pgmajfault": 1450, "pgfault": 987654321}


def test_diskstats_auto_keeps_whole_disks_only():
    d = sources.parse_diskstats(DISKSTATS)
    assert list(d) == ["nvme0n1"]
    assert d["nvme0n1"]["rd_sectors"] == 847031030
    assert d["nvme0n1"]["wr_sectors"] == 485704138
    assert d["nvme0n1"]["io_ticks"] == 2184855
    assert d["nvme0n1"]["in_flight"] == 0


def test_diskstats_explicit_device():
    assert sources.parse_diskstats(DISKSTATS, ["nvme0n1p1"])["nvme0n1p1"]["rd_sectors"] == 800


def test_cpu_stat():
    c = sources.parse_cpu_stat(CPUSTAT)
    assert c["user"] == 100 and c["iowait"] == 50
    assert sources.parse_cpu_stat("") == {}


def test_pid_stat_with_colons_in_comm():
    s = sources.parse_pid_stat(PIDSTAT)
    assert s["comm"] == "VLLM::Worker_TP0"
    assert (s["utime"], s["stime"], s["num_threads"], s["rss_pages"]) == (123456, 6543, 89, 983046)
    assert sources.parse_pid_status_ctxt(PIDSTATUS) == {"vcsw": 123, "nvcsw": 45}
    assert sources.parse_pid_stat("garbage") == {}


def test_vllm_metrics():
    m = sources.parse_vllm_metrics(vllm_metrics_text(123456, 1))
    assert m["steps"] == 123456.0
    assert m["step_tokens"] == 987654.0
    assert m["gen_tokens"] == 128350.0
    assert m["prompt_tokens"] == 5.6e7
    assert m["running"] == 1.0
    assert abs(m["kv_cache_usage"] - 0.0012) < 1e-9
    assert m["accepted_per_pos"] == [26167.0, 22962.0, 19768.0]
    assert m["prompt_by_source"]["local_cache_hit"] == 5.5909376e7
    assert "e2e" not in m
    assert sources.parse_vllm_metrics("") == {}


def test_ib_rate():
    assert sources.parse_ib_rate_mbps("200 Gb/sec (2X NDR)") == 200000
    assert sources.parse_ib_rate_mbps("40 Gb/sec (4X QDR)") == 40000
    assert sources.parse_ib_rate_mbps("") is None


def test_list_ib_ports_and_counters(fake_roots):
    sysr, _ = fake_roots
    ports = sources.list_ib_ports(sysr)
    assert [(p["name"], p["port"], p["state"]) for p in ports] == [
        ("roceP2p1s0f0", 1, "ACTIVE"), ("rocep1s0f0", 1, "ACTIVE"), ("rocep1s0f1", 1, "DOWN")]
    c = sources.read_ib_counters(sysr, "rocep1s0f0", 1)
    assert c["rx_bytes"] == 4000  # ×4
    assert c["tx_bytes"] == 1000
    assert c["rx_pkts"] == 7 and c["rx_write_req"] == 99
    assert "packet_seq_err" not in c  # missing file → absent, not 0
    assert sources.read_ib_fast(sysr, "rocep1s0f0", 1) == (4000, 1000, 7, 8, 0)


def test_missing_ib_tree(tmp_path):
    assert sources.list_ib_ports(tmp_path / "nope") == []
    assert sources.read_ib_fast(tmp_path / "nope", "x", 1) == (0, 0, 0, 0, 0)


def test_block_stat(fake_roots):
    sysr, _ = fake_roots
    assert sources.read_block_stat(sysr, "nvme0n1") == (847031030, 485704138, 2184855)
    assert sources.read_block_stat(sysr, "nope") == (0, 0, 0)


def test_net_ifaces_physical_only(fake_roots):
    sysr, _ = fake_roots
    assert sources.list_net_ifaces(sysr) == ["enp1s0f0np0"]
    assert sources.read_net_stats(sysr, "enp1s0f0np0")["rx_bytes"] == 268013599
    assert sources.list_net_ifaces(sysr, physical_only=False) == ["enp1s0f0np0"]  # lo/docker/wg filtered by name


def test_find_procs_and_sample(fake_roots):
    _, procr = fake_roots
    assert sources.find_procs(procr) == {306134: "VLLM::Worker_TP0", 305388: "vllm"}  # tokentrace itself excluded
    s = sources.read_proc_sample(procr, 306134)
    assert s["comm"] == "VLLM::Worker_TP0" and s["vcsw"] == 123
    assert sources.read_proc_sample(procr, 99999) == {}
    assert sources.find_procs(Path("/definitely/not/there")) == {}


def test_read_helpers(tmp_path):
    f = tmp_path / "v"
    f.write_text(" 42 extra\n")
    assert sources.read_int(f) == 42
    f.write_text("nan\n")
    assert sources.read_int(f, 7) == 7
    assert sources.read_int(tmp_path / "missing") is None
    assert sources.read_text(tmp_path / "missing", "d") == "d"
