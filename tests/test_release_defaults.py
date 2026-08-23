from pathlib import Path

import yaml

from src.stage3.tdd_engine import TDDEngine


ROOT = Path(__file__).resolve().parents[1]


class _LLM:
    pass


class _DockerUnavailable:
    def is_docker_available(self):
        return False


def test_committed_release_defaults_fail_closed() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert config["stage3"]["save_unvalidated"] is False
    assert config["stage3"]["static_validation"] is True


def test_generated_static_code_is_not_release_eligible(monkeypatch) -> None:
    engine = TDDEngine(
        _LLM(),
        _DockerUnavailable(),
        {"skip_docker": True, "static_validation": True},
    )
    monkeypatch.setattr(engine, "_generate_test", lambda *_: "def test_x(): pass")
    monkeypatch.setattr(engine, "_generate_implementation", lambda *_: "def x(): return 1")
    monkeypatch.setattr(
        engine,
        "_validate_static",
        lambda *_: {"status": "syntax_valid", "errors": []},
    )
    rpg = type(
        "RPG",
        (),
        {"graph": type("Graph", (), {"nodes": {"x": {"name": "x"}}})()},
    )()

    eligible, result = engine.generate(rpg, "x")

    assert eligible is False
    assert result["status"] == "syntax_valid"
