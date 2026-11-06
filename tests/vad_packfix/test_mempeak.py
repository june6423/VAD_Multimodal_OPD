"""Tests of scripts/vad_packfix/mempeak.py."""

from scripts.vad_packfix import mempeak

LOG = """container cgroup /sys/fs/cgroup/system.slice/x.scope memory.max 161061273600
09:00:00 anon 50.0 file 400.0 ctr_cur 60.0 ctr_anon 50.0 ctr_file 9.0 ctr_shmem 8.0 ctr_file_mapped 7.0 ctr_kernel 1.0 ctr_oom_kill 9 devshm_used 8.0 avail 2000.0 0 MiB, 0 %
  rss 40.0 pid 1 ray::WorkerDict
09:00:05 anon 80.0 file 400.0 ctr_cur 140.5 ctr_anon 120.0 ctr_file 19.0 ctr_shmem 10.0 ctr_file_mapped 7.5 ctr_kernel 1.5 ctr_oom_kill 9 devshm_used 10.0 avail 1990.0 1000 MiB, 50 %
09:00:10 anon 30.0 file 400.0 ctr_cur 20.0 ctr_anon 12.0 ctr_file 7.0 ctr_shmem 1.0 ctr_file_mapped 1.0 ctr_kernel 1.0 ctr_oom_kill 10 devshm_used 1.0 avail 2100.0 0 MiB, 0 %
09:00:15 anon 30.0 file 400.0 avail 2100.0
"""


def test_mempeak_takes_the_largest_container_charge(tmp_path):
    path = tmp_path / "memwatch.log"
    path.write_text(LOG)
    s = mempeak.summary(mempeak.parse(str(path)))
    assert s["samples"] == 3
    assert s["peak_ctr_cur_sample"]["time"] == "09:00:05"
    assert s["peak_ctr_cur_sample"]["ctr_anon"] == 120.0
    assert s["max"]["ctr_shmem"] == 10.0
    assert (s["oom_kill_first"], s["oom_kill_last"]) == (9.0, 10.0)
