import json

from src.core.rpg import NodeType, RepositoryPlanningGraph
from src.stage3.release_gate import ReleaseGate


def _rpg_with_artifact(root):
    rpg = RepositoryPlanningGraph("receipt")
    node_id = rpg.add_node(
        "hello", NodeType.LEAF, node_id="hello", file_path="src/hello.py"
    )
    artifact = root / "src" / "hello.py"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("def hello():\n    return 'world'\n")
    return rpg, node_id, artifact


def test_validated_repository_gets_verifiable_receipt(tmp_path):
    rpg, node_id, _artifact = _rpg_with_artifact(tmp_path)
    gate = ReleaseGate(tmp_path)
    receipt = gate.evaluate(
        rpg,
        {
            node_id: {
                "implementation": "def hello(): return 'world'",
                "status": "validated",
                "validation_method": "docker",
                "attempts": 1,
            }
        },
        integration_success=True,
        router_stats={"api_calls": 2, "total_tokens": 300, "total_cost": 0.1},
        config={"stage3": {"save_unvalidated": False}},
    )
    path = gate.write(receipt)

    assert receipt["release_eligible"] is True
    assert path.is_file()
    assert gate.verify() is True
    persisted = json.loads(path.read_text())
    assert persisted["usage"]["total_tokens"] == 300
    assert persisted["artifacts"][0]["path"] == "src/hello.py"


def test_unvalidated_or_failed_integration_is_denied(tmp_path):
    rpg, node_id, _artifact = _rpg_with_artifact(tmp_path)
    receipt = ReleaseGate(tmp_path).evaluate(
        rpg,
        {
            node_id: {
                "implementation": "def hello(): return 'world'",
                "status": "syntax_valid",
                "validation_method": "static",
            }
        },
        integration_success=False,
        router_stats={},
        config={},
    )

    assert receipt["release_eligible"] is False
    assert "integration tests did not pass" in receipt["reasons"]
    assert any("not validated" in reason for reason in receipt["reasons"])


def test_artifact_tampering_invalidates_receipt(tmp_path):
    rpg, node_id, artifact = _rpg_with_artifact(tmp_path)
    gate = ReleaseGate(tmp_path)
    receipt = gate.evaluate(
        rpg,
        {node_id: {"implementation": "x", "status": "validated"}},
        integration_success=True,
        router_stats={},
        config={},
    )
    gate.write(receipt)
    artifact.write_text("tampered\n")

    assert gate.verify() is False
