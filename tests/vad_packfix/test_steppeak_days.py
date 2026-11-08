"""Tests of scripts/vad_packfix/steppeak_days.py (DT ANSWER on REPORT-VLM-VAD-PACKFIX-001-002 item 3, VT O3)."""

import json

from scripts.vad_packfix import steppeak_days


def _sample(t, anon, shmem, oom=14, agent=1.0):
    return (f"{t} anon 1.0 file 2.0 ctr_cur {anon + shmem + 5} ctr_anon {anon} ctr_file 5.0 ctr_kernel 0.9 "
            f"ctr_shmem {shmem} ctr_file_mapped 1.0 ctr_oom_kill {oom} devshm_used 0.0 avail 9.0 0 MiB, 0 %\n"
            f"  rss 1.0 pid 1 ray::WorkerDict\n"
            f"  groups ray::WorkerDict 30.0/1 ray::AgentLoopW {agent}/8 other 1.0/9\n")


def _root(tmp_path, memwatch, run_packfix, consoles):
    (tmp_path / "RUN_BOTH.txt").write_text("== date 2026-11-08T23:00:00Z git HEAD x PILOT 0\n")
    (tmp_path / "memwatch.log").write_text("container cgroup x memory.max 1\n" + memwatch)
    arm = tmp_path / "B"
    arm.mkdir()
    (arm / "RUN_PACKFIX.txt").write_text("== date 2026-11-08T23:00:05Z git HEAD x ARM B\n" + run_packfix)
    for name, text in consoles.items():
        (arm / f"console_{name}.log").write_text(text)


def test_a_window_across_midnight_gets_its_samples(tmp_path):
    _root(tmp_path,
          _sample("23:58:50", 50.0, 2.0) + _sample("23:59:59", 90.0, 2.0) + _sample("00:00:30", 94.8, 2.3, agent=17.0)
          + _sample("00:06:00", 10.0, 0.0),
          "== validate val3: step 3 trainer.resume_mode=resume_path x start 2026-11-08T23:58:41Z\n"
          "validate val3 exit 0 wall_seconds 429 dump x\n",
          {"val3": "(TaskRunner pid=9) Resuming from /o/checkpoints/global_step_3\n"
                   "(TaskRunner pid=9) VAD_HOSTMEM pid=9 step_log step 3 utc 00:05:50\n"})
    assert steppeak_days.main(str(tmp_path), str(tmp_path / "S.json")) == 0
    (row,) = json.loads((tmp_path / "S.json").read_text())["rows"]
    assert (row["run"], row["step"], row["wall_seconds"], row["samples"]) == ("val3", 3, 429, 3)
    assert (row["start"], row["end"]) == ("2026-11-08T23:58:41Z", "2026-11-09T00:05:50Z")
    assert row["anon_shmem"] == 97.1 and row["anon_shmem_peak"]["time"] == "2026-11-09T00:00:30Z"
    assert row["largest_group"] == ["ray::WorkerDict", 30.0]


def test_a_time_of_day_that_repeats_after_24_h_counts_only_once(tmp_path):
    # Day 1 at 23:30 has a high sample. The window on day 2 at 23:30 must not get it.
    _root(tmp_path,
          _sample("23:30:10", 140.0, 5.0) + _sample("12:00:00", 60.0, 1.0) + _sample("23:30:10", 70.0, 1.0, agent=40.0),
          "== validate val20_p0: step 20 x start 2026-11-09T23:30:00Z\n"
          "validate val20_p0 exit 0 wall_seconds 60 dump x\n",
          {"val20_p0": "(TaskRunner pid=9) VAD_HOSTMEM pid=9 step_log step 20 utc 23:31:00\n"})
    assert steppeak_days.main(str(tmp_path), str(tmp_path / "S.json")) == 0
    result = json.loads((tmp_path / "S.json").read_text())
    (row,) = result["rows"]
    assert (row["samples"], row["anon_shmem"]) == (1, 71.0)
    assert row["largest_group"] == ["ray::AgentLoopW", 40.0]
    assert result["max_anon_shmem"] == 71.0


def test_copies_and_step_windows_follow_the_day(tmp_path):
    _root(tmp_path,
          _sample("23:59:00", 80.0, 20.0) + _sample("00:01:00", 100.0, 21.0, oom=15),
          "== train full: total 20 save 10 metrics m start 2026-11-08T23:58:00Z\n",
          {"full": "(WorkerDict pid=7) VAD_HOSTMEM pid=7 copy offload_model seconds 1.50 gib 16.91 utc 23:58:30\n"
                   "(TaskRunner pid=8) VAD_HOSTMEM pid=8 step_log step 1 utc 23:59:30\n"
                   "(WorkerDict pid=7) VAD_HOSTMEM pid=7 copy load_model seconds 2.00 gib 16.91 utc 00:00:10\n"
                   "(TaskRunner pid=8) VAD_HOSTMEM pid=8 step_log step 2 utc 00:01:30\n"})
    assert steppeak_days.main(str(tmp_path), str(tmp_path / "S.json")) == 0
    s1, s2 = json.loads((tmp_path / "S.json").read_text())["rows"]
    assert (s1["step"], s1["anon_shmem"], s1["copies"]) == (1, 100.0, {"offload_model": {"count": 1, "seconds": 1.5}})
    assert (s2["step"], s2["wall_seconds"], s2["anon_shmem"]) == (2, 120, 121.0)
    assert s2["copies"] == {"load_model": {"count": 1, "seconds": 2.0}}
    assert (s2["oom_kill_first"], s2["oom_kill_last"]) == (15.0, 15.0)
