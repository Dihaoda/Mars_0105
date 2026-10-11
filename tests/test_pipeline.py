"""Default-size software integration fixture; no dataset/model is downloaded."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from work2.cli import poison_partitions
from work2.data import build_partitions


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def records(domain, size):
            return [{"sample_id": f"fixture-{domain}-{i}", "product_id": f"fixture-product-{domain}-{i}",
                     "domain": domain, "text": f"Unit test fixture text {domain} {i}.", "label": i % 2}
                    for i in range(size)]
        cls.partitions = build_partitions(records("ordinary", 15900), records("heterogeneous", 3200), 2026)

    def test_full_default_protocol_backdoor_and_validation_scope(self):
        result = poison_partitions(self.partitions, "text_backdoor", 2026)
        self.assertNotIn("manifest", result)
        self.assertTrue(result["input_clean_partition_validation"]["passed"])
        self.assertIn("Input clean partitions only", result["validation_scope"])
        self.assertEqual(sum(m["poisoned_count"] for m in result["attack_manifests"].values()), 300)
        for client_id, records in result["clients"].items():
            self.assertEqual(len(records), 500)
            positives = sum(row["label"] for row in records)
            self.assertEqual(positives, 300 if int(client_id) >= 24 else 250)
        self.assertEqual(result["tests"], self.partitions["tests"])
        self.assertEqual(result["root"], self.partitions["root"])
        self.assertFalse(result["training_executed"])

    def test_partition_seed_must_match_the_paired_run(self):
        with self.assertRaisesRegex(ValueError, "requested seed"):
            poison_partitions(self.partitions, "none", 2027)

    def test_changed_clean_records_fail_their_manifest_hash(self):
        tampered = copy.deepcopy(self.partitions)
        tampered["clients"][0][0]["text"] = "A unique changed test fixture string."
        with self.assertRaisesRegex(ValueError, "record hash"):
            poison_partitions(tampered, "none", 2026)


if __name__ == "__main__":
    unittest.main()
