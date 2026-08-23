# Validation status

BLUEPRINT is an alpha research prototype. Its three-stage planning and code
generation pipeline is implemented, but the repository does not yet publish a
reproducible corpus supporting the historical time, cost, or quality figures
in the long-form README.

## Artifact meanings

- **Draft:** code was generated, or only parsed by static syntax checks.
- **Validated:** generated tests executed successfully in Docker for a node.
- **Release:** reserved for a repository whose clean build and integration
  test suite pass. The release-attestation feature will automate this boundary.

`stage3.save_unvalidated` defaults to `false`. Setting it to `true` is an
explicit experimental escape hatch and must never be represented as a
validated release.

## Evidence still required

- pinned prompt/model/provider/config hashes;
- real token and cost accounting from every router;
- clean-container build and integration logs;
- test pass rate and coverage;
- security, dependency, secret, and license checks;
- reproducible golden-project benchmark receipts.

CI currently gates the maintained RPG and fail-closed release tests. The older
`test_phase0_foundation.py` suite targets superseded `CostTracker`,
`GraphOperations`, and `GraphPersistence` APIs and is intentionally not counted
as passing evidence until it is migrated or removed.
