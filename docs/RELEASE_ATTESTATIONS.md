# Validation receipts

Every Stage 3 output now contains `.blueprint/validation-receipt.json`. The
receipt records release eligibility, integration-test status, validation status
for every generated node, an artifact manifest, the effective configuration
hash, and the router's measured API calls, tokens, and cost.

The gate is fail-closed. A repository is release-eligible only when it contains
generated code, every generated node has `status: validated`, all expected code
files exist, and integration tests passed. Otherwise the output remains a draft
and `generate_repository()` returns `False` with explicit denial reasons.

Verify a receipt and detect artifact tampering:

```python
from src.stage3.release_gate import ReleaseGate

assert ReleaseGate("generated_repo").verify()
```

The receipt uses SHA-256 checksums for integrity and reproducibility; it is not
an identity signature. A future release pipeline may sign the receipt digest
with Sigstore without changing the artifact manifest format.
