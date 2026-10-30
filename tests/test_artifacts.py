import json

import numpy as np

from counterworld.artifacts import ArtifactWriter, canonical_json, file_hash


def test_artifact_manifest_hashes_configuration_and_files(tmp_path):
    writer = ArtifactWriter(tmp_path / "run")
    writer.write_jsonl("scores.jsonl", [{"x": np.float64(1.5), "valid": np.array([True, False])}])
    writer.write_json("summary.json", {"infinite": np.inf})
    manifest_path = writer.finalize({"seed": 7})
    manifest = json.loads(manifest_path.read_text())
    assert manifest["protocol"] == "CW-0.2-draft"
    assert manifest["configuration"]["seed"] == 7
    assert manifest["files"]["summary.json"]["sha256"]
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["infinite"] == "Infinity"
    assert len(manifest["files"]["scores.jsonl"]["sha256"]) == 64
    assert file_hash(manifest_path) == file_hash(manifest_path)


def test_canonical_json_is_stable():
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
