"""Tests of scripts/vad_view2/steppeak.py (DIRECTIVE-VLM-VAD-VIEW-002-007: a resumed DIAG_STEPS run in one arm log)."""

import json

from scripts.vad_view2 import steppeak


def _sample(t, anon, shmem, agent=1.0):
    return (f"{t} anon 1.0 file 2.0 ctr_cur {anon + shmem + 5} ctr_anon {anon} ctr_file 5.0 ctr_kernel 0.9 "
            f"ctr_shmem {shmem} ctr_file_mapped 1.0 ctr_oom_kill 17 devshm_used 3.5 avail 9.0 0 MiB, 0 %\n"
            f"  rss 1.0 pid 1 ray::WorkerDict\n"
            f"  groups ray::WorkerDict 30.0/1 ray::AgentLoopW {agent}/8 ray::TaskRunner 9.0/9 other 1.0/9\n")


def test_runs_after_the_run_all_date_count_and_earlier_runs_do_not(tmp_path):
    (tmp_path / "RUN_ALL.txt").write_text("== date 2026-11-17T10:00:00Z full-size steps\n")
    (tmp_path / "memwatch.log").write_text("container cgroup x memory.max 1\n" + _sample("09:30:30", 140.0, 5.0)
                                           + _sample("10:01:30", 100.0, 20.0, agent=40.0)
                                           + _sample("10:31:30", 110.0, 20.0, agent=50.0))
    arm = tmp_path / "T8"
    arm.mkdir()
    (arm / "RUN_VIEW2.txt").write_text(
        "== date 2026-11-17T09:30:00Z code x ARM T8 DIAG_STEPS 1\n"
        "== train old: total 1 save 1 rollout.n 8 metrics m start 2026-11-17T09:30:00Z\n"
        "== date 2026-11-17T10:00:05Z code x ARM T8 DIAG_STEPS 5\n"
        "== train diag5: total 5 save 5 rollout.n 8 metrics m start 2026-11-17T10:00:10Z\n"
        "train diag5 exit 0 wall_seconds 1200 latest 5\n"
        "== date 2026-11-17T10:30:00Z code x ARM T8 DIAG_STEPS 6\n"
        "== train diag6: total 6 save 6 rollout.n 8 metrics m start 2026-11-17T10:30:10Z\n"
        "train diag6 exit 0 wall_seconds 600 latest 6\n")
    (arm / "console_old.log").write_text("(TaskRunner pid=8) VAD_HOSTMEM pid=8 step_log step 1 utc 09:31:00\n")
    (arm / "console_diag5.log").write_text("(TaskRunner pid=8) VAD_HOSTMEM pid=8 step_log step 5 utc 10:02:00\n")
    (arm / "console_diag6.log").write_text("(TaskRunner pid=8) VAD_HOSTMEM pid=8 step_log step 6 utc 10:32:00\n")
    assert steppeak.main(str(tmp_path), str(tmp_path / "S.json")) == 0
    result = json.loads((tmp_path / "S.json").read_text())
    assert [(r["run"], r["step"], r["anon_shmem"]) for r in result["rows"]] == [("diag5", 5, 120.0), ("diag6", 6, 130.0)]
    assert result["max_anon_shmem"] == 130.0
    assert result["rows"][1]["peak_groups"]["AgentLoopW"] == 50.0
    assert result["rows"][1]["peak_groups"]["object_store_devshm"] == 3.5
    table = (tmp_path / "S_GROUPS.tsv").read_text().splitlines()
    assert len(table) == 3 and table[2].split("\t")[:3] == ["T8", "diag6", "6"]
