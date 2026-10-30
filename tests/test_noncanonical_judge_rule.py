from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPAIR = Path(__file__).resolve().parents[1] / "evaluation" / "vision_opd_official" / "repair_judges.py"


def test_noncanonical_labels_count_as_incorrect_without_judge_call(tmp_path: Path) -> None:
    # A fake openai package writes a marker file when a client is created.
    fake_root = tmp_path / "fake_modules"
    (fake_root / "openai").mkdir(parents=True)
    marker = tmp_path / "openai_client_created"
    (fake_root / "openai" / "__init__.py").write_text(
        "from pathlib import Path\n\n\n"
        "class OpenAI:\n"
        "    def __init__(self, *args, **kwargs):\n"
        f"        Path({str(marker)!r}).write_text('created')\n",
        encoding="utf-8",
    )
    labels = ["Yes", "yes.", "No", "no", "", "The response matches the answer."]
    judge_json = tmp_path / "judge.json"
    judge_json.write_text(
        json.dumps([{"sample_uid": f"vstar:question_id:{i}", "judge": label} for i, label in enumerate(labels)]),
        encoding="utf-8",
    )

    # The arguments of scripts/evaluate_official.sh lines 194-199.
    completed = subprocess.run(
        [
            sys.executable,
            str(REPAIR),
            "--judge-json",
            str(judge_json),
            "--api-base",
            "http://127.0.0.1:9/v1",
            "--api-key",
            "EMPTY",
            "--judge-model",
            "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            "--judge-max-tokens",
            "2048",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(fake_root), os.environ.get("PYTHONPATH", "")])},
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()
    assert json.loads(completed.stdout) == {"initial_noncanonical": 4, "canonicalized": 2, "as_incorrect": 2}
    rows = json.loads(judge_json.read_text(encoding="utf-8"))
    assert [row["judge"] for row in rows] == ["Yes", "Yes", "No", "No", "No", "No"]
    assert [row.get("judge_source") for row in rows] == [
        None,
        "llm_canonicalized",
        None,
        "llm_canonicalized",
        "noncanonical_as_incorrect",
        "noncanonical_as_incorrect",
    ]
    assert [row.get("judge_raw_before_repair") for row in rows] == [
        None,
        "yes.",
        None,
        "no",
        "",
        "The response matches the answer.",
    ]
