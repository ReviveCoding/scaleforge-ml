from pathlib import Path

from scaleforge.protocol import canonical_sha256, sha256_file


def test_hashes_are_deterministic(tmp_path: Path) -> None:
    artifact = tmp_path / "evidence.txt"
    artifact.write_text("evidence", encoding="utf-8")
    assert len(sha256_file(artifact)) == 64
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})
