"""Tests of scripts/vad_packfix/steppeak.py."""

import json

from scripts.vad_packfix import steppeak


def _sample(t, cur, anon, shmem, oom):
    return (f"{t} anon 1.0 file 2.0 ctr_cur {cur} ctr_anon {anon} ctr_file {shmem} ctr_kernel 0.9 "
            f"ctr_shmem {shmem} ctr_file_mapped {shmem} ctr_oom_kill {oom} devshm_used 0.0 avail 9.0 0 MiB, 0 %\n"
            f"  rss 1.0 pid 1 ray::WorkerDict\n")


def test_steppeak_assigns_samples_and_copies_to_step_windows(tmp_path):
    (tmp_path / "memwatch.log").write_text(
        "container cgroup x memory.max 1\n"
        + _sample("10:00:00", 5.0, 4.0, 1.0, 11) + _sample("10:00:10", 50.0, 30.0, 20.0, 11)
        + _sample("10:00:30", 120.0, 60.0, 59.0, 11) + _sample("10:00:50", 90.0, 50.0, 39.0, 12))
    arm = tmp_path / "F"
    arm.mkdir()
    (arm / "RUN_PACKFIX.txt").write_text(
        "== train run1: total 1 save 1 val_before_train True metrics m start 2026-11-06T10:00:00Z\n")
    (arm / "console_run1.log").write_text(
        "(WorkerDict pid=7) VAD_HOSTMEM pid=7 copy offload_model seconds 1.50 gib 16.91 utc 10:00:05\n"
        "(TaskRunner pid=8) VAD_HOSTMEM pid=8 step_log step 0 utc 10:00:20\n"
        "(WorkerDict pid=7) VAD_HOSTMEM pid=7 copy offload_optimizer seconds 4.25 gib 33.82 utc 10:00:40\n"
        "(WorkerDict pid=7) VAD_HOSTMEM pid=7 copy offload_model seconds 1.25 gib 16.91 utc 10:00:41\n"
        "(TaskRunner pid=8) VAD_HOSTMEM pid=8 step_log step 1 utc 10:00:55\n")
    assert steppeak.main(str(tmp_path), str(tmp_path / "S.json")) == 0
    result = json.loads((tmp_path / "S.json").read_text())
    step0, step1 = result["rows"]
    assert (step0["step"], step0["ctr_cur"], step0["wall_seconds"]) == (0, 50.0, 20)
    assert step0["copies"] == {"offload_model": {"count": 1, "seconds": 1.5}}
    assert (step1["step"], step1["ctr_cur"], step1["ctr_anon"], step1["ctr_shmem"]) == (1, 120.0, 60.0, 59.0)
    assert (step1["oom_kill_first"], step1["oom_kill_last"]) == (11, 12)
    assert step1["copies"]["offload_optimizer"] == {"count": 1, "seconds": 4.25}
    assert result["max_ctr_cur"] == 120.0


def test_steppeak_gives_the_anon_shmem_peak_and_the_window_of_a_failed_step(tmp_path):
    # DT ANSWER on TASK-VLM-VAD-PACKFIX-001-022: the 134 GiB threshold applies to anon + shmem, and the step that
    # failed (no step_log line) also gets a window up to the end of its run.
    (tmp_path / "memwatch.log").write_text(
        _sample("10:00:00", 5.0, 4.0, 1.0, 11) + _sample("10:00:10", 140.0, 60.0, 10.0, 11)
        + _sample("10:00:20", 100.0, 80.0, 15.0, 11) + _sample("10:00:40", 130.0, 120.0, 8.0, 11)
        + _sample("10:01:30", 1.0, 1.0, 0.0, 11))
    arm = tmp_path / "F"
    arm.mkdir()
    (arm / "RUN_PACKFIX.txt").write_text(
        "== train run1: total 2 save 2 val_before_train True metrics m start 2026-11-06T09:00:00Z\n"
        "train run1 exit 0 wall_seconds 900 latest 2\n"  # a successful run gets no incomplete window
        "== train run2: total 3 save 2 val_before_train False metrics m start 2026-11-06T10:00:00Z\n"
        "train run2 exit 1 wall_seconds 50 latest 2\n")
    (arm / "console_run1.log").write_text("(TaskRunner pid=8) VAD_HOSTMEM pid=8 step_log step 2 utc 09:14:00\n")
    (arm / "console_run2.log").write_text("(TaskRunner pid=8) VAD_HOSTMEM pid=8 step_log step 3 utc 10:00:25\n")
    assert steppeak.main(str(tmp_path), str(tmp_path / "S.json")) == 0
    result = json.loads((tmp_path / "S.json").read_text())
    run1, done, failed = result["rows"]
    assert (run1["run"], run1["step"], run1["complete"], run1["samples"]) == ("run1", 2, True, 0)
    assert (done["step"], done["complete"], done["ctr_cur"], done["anon_shmem"]) == (3, True, 140.0, 95.0)
    assert done["anon_shmem_peak"] == {"time": "10:00:20", "ctr_cur": 100.0, "ctr_anon": 80.0, "ctr_shmem": 15.0,
                                       "ctr_file": 15.0}
    assert (failed["step"], failed["complete"], failed["start"], failed["end"]) == (4, False, "10:00:25", "10:00:50")
    assert (failed["samples"], failed["anon_shmem"]) == (1, 128.0)
    assert (result["max_ctr_cur"], result["max_anon_shmem"]) == (140.0, 128.0)
