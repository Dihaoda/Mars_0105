"""Software-only fixtures, unrelated to Table 1 or model training."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from work2.cli import analyze_supplied_run, write_new_json


class CliTests(unittest.TestCase):
    def make_fixture(self, root):
        # Fabricated input solely for exercising an import contract in a unit
        # test. These records are never exported as experimental observations.
        predictions = [
            {"sample_id": f"{domain}-{i}", "domain": domain,
             "y_true": i % 2, "y_pred": i % 2}
            for domain in ("ordinary", "heterogeneous") for i in range(2000)
        ]
        decisions = [
            {"round": r, "client_id": c,
             "role": "heterogeneous" if 18 <= c < 24 else "benign_designated" if c >= 24 else "ordinary",
             "accepted": True, "normalized_weight": 1 / 30}
            for r in range(1, 51) for c in range(30)
        ]
        p, d, m = root / "predictions.jsonl", root / "decisions.jsonl", root / "manifest.json"
        for path, rows in ((p, predictions), (d, decisions)):
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        manifest = {
            "unit_test_fixture": True,
            "data_origin": "measured_training", "measured": True,
            "status": "completed", "completed_rounds": 50,
            "run_id": "unit-test-input-contract", "seed": 2026,
            "attack": "none", "method": "FedAvg", "model_name": "Qwen/Qwen2.5-3B-Instruct",
            "dataset_manifest_sha256": "0" * 64, "protocol_sha256": "0" * 64,
            "predictions_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "decisions_sha256": hashlib.sha256(d.read_bytes()).hexdigest(),
        }
        m.write_text(json.dumps(manifest), encoding="utf-8")
        return p, d, m, manifest

    def test_complete_import_has_explicit_provenance_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            p, d, m, _ = self.make_fixture(Path(directory))
            result = analyze_supplied_run(p, d, m)
            self.assertEqual(result["metrics"]["ordinary_accuracy"], 1)
            self.assertEqual(result["metrics"]["h_adverse_rate"], 0)
            self.assertEqual(result["data_origin"], "computed_from_supplied_observations")
            self.assertIn("not_independently_verified", result["input_provenance"])
            self.assertFalse(result["training_executed_by_this_command"])

    def test_reference_manifest_cannot_be_imported(self):
        with tempfile.TemporaryDirectory() as directory:
            p, d, m, manifest = self.make_fixture(Path(directory))
            manifest.update(data_origin="illustrative_reference", measured=False)
            m.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not workflow examples"):
                analyze_supplied_run(p, d, m)

    def test_tampered_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p, d, m, _ = self.make_fixture(Path(directory))
            p.write_text(p.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                analyze_supplied_run(p, d, m)

    def test_none_condition_cannot_claim_malicious_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            p, d, m, manifest = self.make_fixture(Path(directory))
            d.write_text(d.read_text(encoding="utf-8").replace("benign_designated", "malicious"), encoding="utf-8")
            manifest["decisions_sha256"] = hashlib.sha256(d.read_bytes()).hexdigest()
            m.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "roles do not match"):
                analyze_supplied_run(p, d, m)

    def test_demo_rows_rejected_even_with_mislabelled_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            p, d, m, manifest = self.make_fixture(Path(directory))
            rows = [json.loads(line) for line in p.read_text().splitlines()]
            rows[0]["data_origin"] = "simulated_workflow"
            p.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            manifest["predictions_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            m.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Demonstration records"):
                analyze_supplied_run(p, d, m)

    def test_previous_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_new_json(path, {"existing": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_new_json(path, {"existing": False})
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
