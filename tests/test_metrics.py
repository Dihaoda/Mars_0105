"""Small validation fixtures, not experimental results or table-value inputs."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from work2.metrics import aggregate_run, mean_sample_std


def predictions():
    return [
        {"sample_id": "o0", "domain": "ordinary", "y_true": 0, "y_pred": 0},
        {"sample_id": "o1", "domain": "ordinary", "y_true": 1, "y_pred": 1},
        {"sample_id": "h0", "domain": "heterogeneous", "y_true": 0, "y_pred": 1},
        {"sample_id": "h1", "domain": "heterogeneous", "y_true": 1, "y_pred": 1},
    ]


def decisions():
    return [
        {"round": r, "client_id": i,
         "role": "heterogeneous" if 18 <= i <= 23 else "malicious" if i >= 24 else "ordinary",
         "accepted": True, "normalized_weight": 1 / 30}
        for r in (1, 2) for i in range(30)
    ]


class MetricsTests(unittest.TestCase):
    def run_metrics(self, ps=None, ds=None, method="FedAvg", **kwargs):
        return aggregate_run(
            predictions() if ps is None else ps,
            decisions() if ds is None else ds,
            method, expected_rounds=2, expected_test_size=2, **kwargs,
        )

    def test_valid_fedavg_counts_and_balanced_accuracy(self):
        result = self.run_metrics()
        self.assertEqual(result["ordinary_accuracy"], 1)
        self.assertEqual(result["heterogeneous_accuracy"], 0.5)
        self.assertEqual(result["h_false_rejection_rate"], 0)
        self.assertAlmostEqual(result["h_relative_weight_retention"], 1)
        self.assertEqual(result["h_decision_count"], 12)
        self.assertTrue(result["fedavg_false_rejection_zero_by_construction"])

    def test_truncated_predictions_cannot_be_reported_as_complete(self):
        with self.assertRaisesRegex(ValueError, "test count"):
            self.run_metrics(ps=predictions()[:-1])

    def test_truncated_client_rounds_cannot_be_reported_as_complete(self):
        with self.assertRaisesRegex(ValueError, "decision count"):
            self.run_metrics(ds=decisions()[:-1])

    def test_duplicate_client_round_with_same_total_length_is_rejected(self):
        ds = decisions()
        ds[-1] = deepcopy(ds[0])
        with self.assertRaisesRegex(ValueError, "duplicate client-round"):
            self.run_metrics(ds=ds)

    def test_overlapping_sample_ids_between_domains_are_rejected(self):
        ps = predictions()
        ps[2]["sample_id"] = ps[0]["sample_id"]
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            self.run_metrics(ps=ps)

    def test_unbalanced_truth_is_rejected_even_with_correct_size(self):
        ps = predictions()
        ps[0]["y_true"] = 1
        with self.assertRaisesRegex(ValueError, "balanced"):
            self.run_metrics(ps=ps)

    def test_nan_infinite_negative_and_nonnumeric_weights_are_rejected(self):
        for invalid in (float("nan"), float("inf"), -0.01, "0.03", True):
            with self.subTest(invalid=invalid):
                ds = decisions()
                ds[0]["normalized_weight"] = invalid
                with self.assertRaises(ValueError):
                    self.run_metrics(ds=ds)

    def test_each_round_must_have_normalized_weights(self):
        ds = decisions()
        ds[0]["normalized_weight"] = 0
        with self.assertRaisesRegex(ValueError, "sum to one"):
            self.run_metrics(ds=ds)

    def test_fedavg_cannot_hide_downweighting_behind_all_accepted_flags(self):
        ds = decisions()
        ds[18]["normalized_weight"] -= 1 / 60
        ds[0]["normalized_weight"] += 1 / 60
        # The shares remain normalized and every accepted flag is true, but
        # this is not FedAvg for thirty equal-size client datasets.
        self.assertTrue(all(row["accepted"] for row in ds))
        with self.assertRaisesRegex(ValueError, "must each equal 1/30"):
            self.run_metrics(ds=ds, method="FedAvg")

    def test_uniform_fedavg_tolerates_json_float_rounding(self):
        ds = decisions()
        for row in ds:
            row["normalized_weight"] = round(1 / 30, 14)
        self.assertEqual(self.run_metrics(ds=ds)["h_false_rejection_rate"], 0)

    def test_rejected_client_cannot_keep_positive_weight(self):
        ds = decisions()
        ds[18]["accepted"] = False
        with self.assertRaisesRegex(ValueError, "exactly zero"):
            self.run_metrics(ds=ds, method="IndexGuard")

    def test_hard_filter_and_low_trust_are_distinct(self):
        ds = decisions()
        threshold = 0.25 / 30
        for offset in (0, 30):
            # Client 18 is accepted but downweighted: a FLTrust diagnostic only.
            ds[offset + 18]["normalized_weight"] = threshold / 2
            ds[offset]["normalized_weight"] += 1 / 30 - threshold / 2
        for method in ("IndexGuard", "FLAME"):
            with self.subTest(method=method):
                result = self.run_metrics(ds=ds, method=method)
                self.assertEqual(result["h_false_rejection_rate"], 0)
                self.assertIsNone(result["h_low_trust_suppression_rate"])
        trust = self.run_metrics(ds=ds, method="FLTrust")
        self.assertAlmostEqual(trust["h_low_trust_suppression_rate"], 1 / 6)
        self.assertEqual(trust["h_native_zero_trust_rate"], 0)
        self.assertIsNone(trust["h_false_rejection_rate"])
        self.assertEqual(trust["h_adverse_metric"], "low_trust_suppression_rate")

    def test_low_trust_threshold_is_strict_less_than(self):
        ds = decisions()
        threshold = 0.25 / 30
        ds[18]["normalized_weight"] = threshold
        ds[0]["normalized_weight"] += 1 / 30 - threshold
        self.assertEqual(self.run_metrics(ds=ds, method="FLTrust")["h_adverse_rate"], 0)

    def test_hard_filter_rejection_denominator_and_fedavg_contract(self):
        ds = decisions()
        ds[18]["accepted"] = False
        ds[18]["normalized_weight"] = 0
        ds[0]["normalized_weight"] += 1 / 30
        for method in ("IndexGuard", "FLAME"):
            with self.subTest(method=method):
                result = self.run_metrics(ds=ds, method=method)
                self.assertEqual(result["h_adverse_count"], 1)
                self.assertAlmostEqual(result["h_false_rejection_rate"], 1 / 12)
        with self.assertRaisesRegex(ValueError, "accept every client"):
            self.run_metrics(ds=ds)

    def test_role_truth_must_match_designated_ids_and_stay_fixed(self):
        ds = decisions()
        ds[18]["role"] = "malicious"
        with self.assertRaisesRegex(ValueError, "inconsistent role"):
            self.run_metrics(ds=ds)
        ds = decisions()
        ds[54]["role"] = "benign_designated"
        with self.assertRaisesRegex(ValueError, "changed"):
            self.run_metrics(ds=ds)

    def test_no_attack_designated_benign_roles_are_valid(self):
        ds = decisions()
        for row in ds:
            if row["client_id"] >= 24:
                row["role"] = "benign_designated"
        self.assertEqual(self.run_metrics(ds=ds)["h_adverse_rate"], 0)

    def test_invalid_round_label_and_boolean_acceptance_are_rejected(self):
        ds = decisions()
        ds[0]["round"] = 0
        with self.assertRaisesRegex(ValueError, "round"):
            self.run_metrics(ds=ds)
        ds = decisions()
        ds[0]["accepted"] = 1
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.run_metrics(ds=ds)
        ps = predictions()
        ps[0]["y_pred"] = True
        with self.assertRaisesRegex(ValueError, "integer 0 or 1"):
            self.run_metrics(ps=ps)

    def test_sample_standard_deviation_is_over_runs_and_not_population_sd(self):
        result = mean_sample_std([0.7, 0.8, 0.9])
        self.assertAlmostEqual(result["mean"], 0.8)
        self.assertAlmostEqual(result["sample_std"], 0.1)
        self.assertEqual(result["n"], 3)
        self.assertIsNone(mean_sample_std([0.8])["sample_std"])
        with self.assertRaises(ValueError):
            mean_sample_std([])
        with self.assertRaises(ValueError):
            mean_sample_std([0.8, float("nan")])


if __name__ == "__main__":
    unittest.main()
