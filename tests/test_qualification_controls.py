import json
import subprocess
from pathlib import Path

import pytest

from scaleforge.protocol import canonical_sha256, sha256_file
from scripts import qualify_model


def test_verify_freeze_rejects_changed_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "parser.py"
    artifact.write_text("frozen", encoding="utf-8")
    config = {"protocol_identity": "SF-MODEL-v1"}
    manifest = {
        "state": "FROZEN",
        "protocol_identity": "SF-MODEL-v1",
        "configuration_sha256": canonical_sha256(config),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "file_sha256": {str(artifact): sha256_file(artifact)},
    }
    qualify_model.verify_freeze(config, manifest)
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen artifact mismatch"):
        qualify_model.verify_freeze(config, manifest)


def test_access_ledger_update_is_idempotent_and_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger_path = tmp_path / "FINAL_ACCESS_LEDGER.json"
    ledger_path.write_text(
        json.dumps({"schema_version": "1.0.0", "updated_at": "initial", "entries": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(qualify_model, "LEDGER_PATH", ledger_path)
    entry = {"access_id": "access-test", "outcome_exposed": False}
    qualify_model.update_ledger(entry)
    qualify_model.update_ledger({**entry, "outcome_exposed": True})
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(ledger["entries"]) == 1
    assert ledger["entries"][0]["outcome_exposed"] is True
    assert not ledger_path.with_suffix(".json.tmp").exists()
