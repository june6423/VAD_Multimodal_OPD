"""VLM-VAD-ROLLOUT-001 gate G8: dp_actor imports with the vzero stub, and each stub function raises."""

import ast
import importlib
from pathlib import Path

import pytest

STUB = Path(__file__).resolve().parents[2] / "verl/trainer/ppo/vzero.py"
NAMES = ("compute_vzero_trajectory_scores", "compute_vzero_evidence_weights")


def test_g8_dp_actor_imports_and_stub_functions_raise():
    dp_actor = importlib.import_module("verl.workers.actor.dp_actor")
    assert hasattr(dp_actor, "DataParallelPPOActor")

    tree = ast.parse(STUB.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert sorted(f.name for f in functions) == sorted(NAMES)
    assert all(len(f.body) == 1 and isinstance(f.body[0], ast.Raise) for f in functions)
    assert [type(node).__name__ for node in tree.body] == ["Expr", "FunctionDef", "FunctionDef"]

    for name in NAMES:
        with pytest.raises(NotImplementedError):
            getattr(dp_actor, name)(1, key=2)
    print(f"G8 import of verl.workers.actor.dp_actor OK; module-level nodes {[type(n).__name__ for n in tree.body]}; "
          f"{', '.join(NAMES)} raise NotImplementedError")
