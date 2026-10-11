"""Contract tests for transparent workflow rehearsal, using no ML dependencies."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from work2.demo import ATTACKS, METHODS, METRIC_KINDS, STEPS, generate_demo


VALUES = [
    (0.0, 93.2, 87.4), (16.0, 92.9, 84.8), (23.7, 93.0, 82.6), (24.5, 92.8, 81.1),
    (0.0, 85.5, 78.4), (29.3, 90.4, 80.3), (38.7, 91.2, 78.1), (41.5, 91.0, 76.8),
    (0.0, 93.0, 85.8), (31.8, 92.4, 79.6), (44.2, 92.0, 76.9), (39.6, 92.2, 78.1),
]


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "project"
        (self.root / "configs").mkdir(parents=True)
        (self.root / "reference").mkdir()
        self.protocol = {"seeds": [2026, 2027, 2028], "federated": {"rounds": 50},
                         "clients": {"total": 30}, "model": {"name": "example-model"}}
        self.reference = {"data_origin": "illustrative_reference", "measured": False,
                          "unit": "percent", "rows": []}
        for (attack, method), values in zip(
                ((a, m) for a in ATTACKS for m in METHODS), VALUES):
            self.reference["rows"].append({
                "attack": attack, "method": method,
                "heterogeneous_misclassification_pct": values[0],
                "ordinary_accuracy_pct": values[1], "heterogeneous_accuracy_pct": values[2],
                "metric_kind": METRIC_KINDS[method],
            })
        self.write_inputs()
        self.output = Path(self.temporary.name) / "output"

    def tearDown(self):
        self.temporary.cleanup()

    def write_inputs(self):
        (self.root / "configs" / "protocol.json").write_text(json.dumps(self.protocol), encoding="utf-8")
        (self.root / "reference" / "table1.json").write_text(json.dumps(self.reference), encoding="utf-8")

    def snapshot(self, directory):
        return {p.relative_to(directory).as_posix(): p.read_bytes()
                for p in directory.rglob("*") if p.is_file()}

    def test_full_plan_and_all_rounds_have_explicit_origin(self):
        summary = generate_demo(self.root, self.output)
        self.assertEqual(summary["round_pipeline_event_count"], 36 * 50)
        plan = json.loads((self.output / "run_plan.json").read_text())
        self.assertEqual(len(plan["runs"]), 36)
        expected_reference = {(r["attack"], r["method"]): r for r in self.reference["rows"]}
        for run in plan["runs"]:
            self.assertEqual(run["status"], "not_executed")
            events = [json.loads(line) for line in
                      (self.output / run["events_path"]).read_text().splitlines()]
            self.assertEqual(len(events), 52)
            self.assertEqual([e["event_index"] for e in events], list(range(52)))
            self.assertEqual(events[0]["event"], "begin")
            self.assertEqual(events[0]["actual_training_status"], "not_executed")
            self.assertEqual(events[0]["seed"], run["seed"])
            self.assertEqual(events[0]["attack"], run["attack"])
            self.assertEqual(events[0]["method"], run["method"])
            self.assertEqual(events[-1]["event"], "end")
            for event in events:
                self.assertEqual(event["data_origin"], "simulated_workflow")
                self.assertIs(event["measured"], False)
                self.assertIs(event["training_executed"], False)
                self.assertFalse(set(event) & {"timestamp", "date", "start_time", "end_time"})
            for number, event in enumerate(events[1:-1], 1):
                self.assertEqual(event["event"], "round_pipeline")
                self.assertEqual(event["round"], number)
                self.assertEqual(tuple(step["name"] for step in event["steps"]), STEPS)
                self.assertTrue(all(s["execution_status"] == "not_executed" for s in event["steps"]))
                self.assertTrue(all(value is None for value in event["measurements"].values()))
                self.assertIsNone(event["client_decisions"])
                self.assertIsNone(event["client_weights"])
                self.assertIsNone(event["checkpoint_artifact"])
                self.assertIs(event["checkpoint_written"], False)
                self.assertNotIn("reference_row", event)
            self.assertEqual(events[-1]["reference_row"], expected_reference[(run["attack"], run["method"])])
            self.assertEqual(events[-1]["reference_row_role"], "shared_illustrative_reference_not_a_seed_measurement")
            self.assertEqual(events[-1]["next_step"]["action"], "run")
        self.assertFalse(any(p.suffix in {".pt", ".bin", ".safetensors", ".ckpt"}
                             for p in self.output.rglob("*")))

    def test_event_logs_are_compact_for_publication(self):
        generate_demo(self.root, self.output)
        logs = list(self.output.glob("runs/*/events.jsonl"))
        self.assertEqual(len(logs), 36)
        for path in logs:
            data = path.read_bytes()
            self.assertLess(len(data), 50_000)
            lines = data.decode("utf-8").splitlines()
            for line in lines:
                self.assertEqual(line, json.dumps(json.loads(line), ensure_ascii=False,
                                                 sort_keys=True, separators=(",", ":")))
            for line in lines[1:-1]:
                event = json.loads(line)
                self.assertNotIn("note", event)
                self.assertNotIn("actual_training_status", event)

    def test_reference_bytes_and_csv_values_are_preserved(self):
        generate_demo(self.root, self.output)
        self.assertEqual((self.output / "reference_table.json").read_bytes(),
                         (self.root / "reference" / "table1.json").read_bytes())
        with (self.output / "reference_table.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 12)
        for actual, expected in zip(rows, self.reference["rows"]):
            self.assertEqual(actual["data_origin"], "illustrative_reference")
            self.assertEqual(actual["measured"], "false")
            for field, value in expected.items():
                self.assertEqual(float(actual[field]) if field.endswith("_pct") else actual[field], value)

    def test_manifest_contains_only_true_generated_file_hashes(self):
        generate_demo(self.root, self.output)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertIn("do not establish that training occurred", manifest["integrity_scope"])
        self.assertIs(manifest["measured"], False)
        for item in manifest["files"]:
            data = (self.output / item["path"]).read_bytes()
            self.assertEqual(item["bytes"], len(data))
            self.assertEqual(item["sha256"], hashlib.sha256(data).hexdigest())
        expected = set(self.snapshot(self.output)) - {"manifest.json"}
        self.assertEqual({item["path"] for item in manifest["files"]}, expected)

    def test_deterministic_and_identical_existing_output_is_accepted(self):
        first = generate_demo(self.root, self.output)
        before = self.snapshot(self.output)
        second = generate_demo(self.root, self.output)
        self.assertEqual(first, second)
        another = self.output.with_name("another")
        generate_demo(self.root, another)
        self.assertEqual(before, self.snapshot(another))
        self.assertEqual(before, self.snapshot(self.output))

    def test_refuses_unmanaged_or_changed_files_without_overwriting(self):
        generate_demo(self.root, self.output)
        unrelated = self.output / "my_notes.txt"
        unrelated.write_text("keep me", encoding="utf-8")
        before = self.snapshot(self.output)
        with self.assertRaises(FileExistsError):
            generate_demo(self.root, self.output)
        self.assertEqual(before, self.snapshot(self.output))
        unrelated.unlink()
        (self.output / "run_index.json").write_text("user change", encoding="utf-8")
        before = self.snapshot(self.output)
        with self.assertRaises(FileExistsError):
            generate_demo(self.root, self.output)
        self.assertEqual(before, self.snapshot(self.output))

    def test_refuses_nonmanaged_empty_subdirectory(self):
        self.output.mkdir()
        (self.output / "user-work").mkdir()
        with self.assertRaises(FileExistsError):
            generate_demo(self.root, self.output)
        self.assertTrue((self.output / "user-work").is_dir())

    def test_invalid_reference_origin_and_scope_are_rejected(self):
        self.reference["measured"] = True
        self.write_inputs()
        with self.assertRaises(ValueError):
            generate_demo(self.root, self.output)
        self.assertFalse(self.output.exists())
        self.reference["measured"] = False
        self.protocol["seeds"] = [2026, 2027, 2029]
        self.write_inputs()
        with self.assertRaises(ValueError):
            generate_demo(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_method_metric_mismatch_is_rejected(self):
        for method in METHODS:
            with self.subTest(method=method):
                row = next(row for row in self.reference["rows"] if row["method"] == method)
                expected = row["metric_kind"]
                row["metric_kind"] = ("heterogeneous_false_rejection" if method in ("FedAvg", "FLTrust")
                                      else "heterogeneous_low_trust_suppression")
                self.write_inputs()
                with self.assertRaisesRegex(ValueError, "Incorrect metric kind"):
                    generate_demo(self.root, self.output)
                self.assertFalse(self.output.exists())
                row["metric_kind"] = expected


if __name__ == "__main__":
    unittest.main()
