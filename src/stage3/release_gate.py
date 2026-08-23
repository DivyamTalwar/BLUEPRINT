"""Fail-closed validation receipts for generated repositories."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.core.rpg import RepositoryPlanningGraph


class ReleaseGate:
    """Evaluate a generated repository and emit a verifiable local receipt."""

    RECEIPT_PATH = Path(".blueprint/validation-receipt.json")
    SCHEMA_VERSION = "1.0"

    def __init__(self, repository_path: str | Path):
        self.repository_path = Path(repository_path).resolve()

    @staticmethod
    def _canonical(value: Any) -> bytes:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _artifact_manifest(self) -> list[Dict[str, Any]]:
        receipt = (self.repository_path / self.RECEIPT_PATH).resolve()
        artifacts = []
        for path in sorted(item for item in self.repository_path.rglob("*") if item.is_file()):
            if path.resolve() == receipt:
                continue
            artifacts.append(
                {
                    "path": path.relative_to(self.repository_path).as_posix(),
                    "sha256": self._sha256(path),
                    "size": path.stat().st_size,
                }
            )
        return artifacts

    def evaluate(
        self,
        rpg: RepositoryPlanningGraph,
        generated_code: Dict[str, Dict[str, Any]],
        *,
        integration_success: bool,
        router_stats: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return a receipt payload; every release condition is fail-closed."""
        reasons = []
        if not generated_code:
            reasons.append("no generated code")
        if not integration_success:
            reasons.append("integration tests did not pass")

        node_results = []
        missing_artifacts = []
        for node_id in sorted(generated_code):
            result = generated_code[node_id]
            status = result.get("status", "unknown")
            validation_method = result.get("validation_method", "none")
            node_results.append(
                {
                    "node_id": node_id,
                    "status": status,
                    "validation_method": validation_method,
                    "attempts": result.get("attempts", 0),
                }
            )
            if status != "validated":
                reasons.append(f"node {node_id} is {status}, not validated")
            node = rpg.graph.nodes[node_id] if node_id in rpg.graph else {}
            relative_path = node.get("file_path") or node.get("parent_file")
            if relative_path and not (self.repository_path / relative_path).is_file():
                missing_artifacts.append(relative_path)

        if missing_artifacts:
            reasons.append(
                "missing generated artifacts: " + ", ".join(sorted(set(missing_artifacts)))
            )

        receipt: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "release_eligible": not reasons,
            "reasons": reasons,
            "configuration_sha256": hashlib.sha256(self._canonical(config)).hexdigest(),
            "usage": {
                "api_calls": router_stats.get("api_calls", 0),
                "total_tokens": router_stats.get("total_tokens", 0),
                "total_cost": router_stats.get("total_cost", 0),
            },
            "validation": {
                "integration_tests_passed": integration_success,
                "nodes": node_results,
            },
            "artifacts": self._artifact_manifest(),
        }
        receipt["receipt_sha256"] = hashlib.sha256(self._canonical(receipt)).hexdigest()
        return receipt

    def write(self, receipt: Dict[str, Any]) -> Path:
        """Atomically write a receipt within the generated repository."""
        destination = self.repository_path / self.RECEIPT_PATH
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
        return destination

    def verify(self) -> bool:
        """Verify both the receipt checksum and every recorded artifact hash."""
        path = self.repository_path / self.RECEIPT_PATH
        try:
            receipt = json.loads(path.read_text())
            claimed = receipt.pop("receipt_sha256")
            actual = hashlib.sha256(self._canonical(receipt)).hexdigest()
            if claimed != actual:
                return False
            for artifact in receipt.get("artifacts", []):
                candidate = (self.repository_path / artifact["path"]).resolve()
                if (
                    not candidate.is_relative_to(self.repository_path)
                    or candidate == path.resolve()
                    or not candidate.is_file()
                    or candidate.stat().st_size != artifact["size"]
                    or self._sha256(candidate) != artifact["sha256"]
                ):
                    return False
            return True
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
