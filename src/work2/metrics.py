"""Validate complete experiment records and calculate method-specific metrics.

This module never generates observations. Its inputs may be real or explicitly
simulated records; callers must preserve that provenance when publishing results.
Rates are returned on a 0--1 scale without rounding. Client-round observations
are descriptive denominators, not independent statistical replicates.
"""

from __future__ import annotations

import math
from numbers import Real
from statistics import mean, stdev
from typing import Iterable


CLIENT_COUNT = 30
DESIGNATED_IDS = frozenset(range(24, 30))
DEFAULT_H_IDS = tuple(range(18, 24))
DOMAINS = ("ordinary", "heterogeneous")
WEIGHT_SUM_ABS_TOLERANCE = 1e-8
UNIFORM_WEIGHT_ABS_TOLERANCE = 1e-12
LOW_TRUST_FRACTION = 0.25


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def mean_sample_std(values: Iterable[float]) -> dict:
    """Summarize independent run-level values; one run has undefined sample SD.

    Call this on one aggregate value per seed, not on individual rounds or
    clients. The function cannot infer whether supplied values are independent.
    """
    numbers = [_finite_number(value, "value") for value in values]
    if not numbers:
        raise ValueError("at least one run-level value is required")
    return {
        "mean": mean(numbers),
        "sample_std": stdev(numbers) if len(numbers) > 1 else None,
        "n": len(numbers),
    }


def aggregate_run(
    predictions: list[dict],
    decisions: list[dict],
    method: str,
    expected_rounds: int = 50,
    h_ids: Iterable[int] = DEFAULT_H_IDS,
    expected_test_size: int = 2000,
) -> dict:
    """Validate a complete run and return accuracy and H-treatment metrics.

    ``predictions`` contains exactly ``expected_test_size`` rows per domain,
    with globally unique ``sample_id`` and balanced binary ``y_true`` labels.
    ``y_true`` and ``y_pred`` must be integer 0/1. Predictions represent a single
    declared checkpoint, normally the final round; repeated checkpoint rows
    cannot be combined here because sample IDs must be globally unique.

    ``decisions`` contains every (round, client_id) pair for rounds 1..R and
    clients 0..29. H clients have role ``heterogeneous``; IDs 24..29 have role
    ``malicious`` or ``benign_designated``; the remainder are ``ordinary``.
    Roles are fixed within a run. ``accepted`` must be a boolean and rejected
    clients must have zero weight. ``normalized_weight`` is the nonnegative
    aggregation selection/trust share whose per-round sum equals one. For
    FLTrust it is normalized ReLU trust, BEFORE root-norm rescaling; the final
    effective update coefficient is not an interchangeable input.

    IndexGuard/FLAME: H adverse event is ``accepted is False``.
    FedAvg: all clients must be accepted and each must have weight 1/30, because
    the protocol gives all clients the same sample count. False rejection is zero
    by construction.
    FLTrust: H adverse event is weight < (1/30)*0.25. This is the declared
    low-trust-suppression diagnostic, NOT an algorithm-native binary FPR.
    """
    expected_rounds = _positive_int(expected_rounds, "expected_rounds")
    expected_test_size = _positive_int(expected_test_size, "expected_test_size")
    if expected_test_size % 2:
        raise ValueError("expected_test_size must be even for balanced labels")
    if not isinstance(method, str):
        raise ValueError("method must name FedAvg, IndexGuard, FLAME, or FLTrust")
    method = method.strip().lower()
    canonical_names = {
        "fedavg": "FedAvg", "indexguard": "IndexGuard",
        "flame": "FLAME", "fltrust": "FLTrust",
    }
    if method not in canonical_names:
        raise ValueError(f"unsupported method: {method!r}")
    h_list = list(h_ids)
    if not h_list or any(type(i) is not int or not 0 <= i < 24 for i in h_list):
        raise ValueError("h_ids must contain distinct benign client IDs in 0..23")
    if len(set(h_list)) != len(h_list):
        raise ValueError("h_ids must not contain duplicates")
    h_set = frozenset(h_list)
    if not isinstance(predictions, list) or not isinstance(decisions, list):
        raise ValueError("predictions and decisions must be lists of records")

    seen_samples: set = set()
    prediction_counts = {domain: 0 for domain in DOMAINS}
    correct_counts = {domain: 0 for domain in DOMAINS}
    label_counts = {domain: {0: 0, 1: 0} for domain in DOMAINS}
    for row in predictions:
        if not isinstance(row, dict):
            raise ValueError("each prediction must be a dictionary")
        required = {"sample_id", "domain", "y_true", "y_pred"}
        if not required.issubset(row):
            raise ValueError(f"prediction missing fields: {sorted(required - row.keys())}")
        sample_id = row["sample_id"]
        if type(sample_id) not in (str, int) or (isinstance(sample_id, str) and not sample_id.strip()):
            raise ValueError("sample_id must be a nonempty string or an integer")
        if sample_id in seen_samples:
            raise ValueError(f"duplicate sample_id, including across domains: {sample_id!r}")
        seen_samples.add(sample_id)
        domain = row["domain"]
        if domain not in DOMAINS:
            raise ValueError(f"unknown prediction domain: {domain!r}")
        for label_field in ("y_true", "y_pred"):
            if type(row[label_field]) is not int or row[label_field] not in (0, 1):
                raise ValueError(f"{label_field} must be integer 0 or 1")
        prediction_counts[domain] += 1
        label_counts[domain][row["y_true"]] += 1
        correct_counts[domain] += row["y_true"] == row["y_pred"]
    for domain in DOMAINS:
        if prediction_counts[domain] != expected_test_size:
            raise ValueError(f"{domain} test count must equal {expected_test_size}, got {prediction_counts[domain]}")
        if label_counts[domain] != {0: expected_test_size // 2, 1: expected_test_size // 2}:
            raise ValueError(f"{domain} true labels must be balanced")

    expected_count = expected_rounds * CLIENT_COUNT
    if len(decisions) != expected_count:
        raise ValueError(f"decision count must equal {expected_count}, got {len(decisions)}")
    seen_pairs: set[tuple[int, int]] = set()
    role_by_client: dict[int, str] = {}
    weights_by_round: dict[int, list[float]] = {r: [] for r in range(1, expected_rounds + 1)}
    h_weights: list[float] = []
    h_rejected = h_zero_trust = h_low_trust = 0
    threshold = LOW_TRUST_FRACTION / CLIENT_COUNT
    for row in decisions:
        if not isinstance(row, dict):
            raise ValueError("each decision must be a dictionary")
        required = {"round", "client_id", "role", "accepted", "normalized_weight"}
        if not required.issubset(row):
            raise ValueError(f"decision missing fields: {sorted(required - row.keys())}")
        round_id, client_id = row["round"], row["client_id"]
        if type(round_id) is not int or not 1 <= round_id <= expected_rounds:
            raise ValueError("decision round is outside the complete expected range")
        if type(client_id) is not int or not 0 <= client_id < CLIENT_COUNT:
            raise ValueError("client_id must be an integer in 0..29")
        pair = (round_id, client_id)
        if pair in seen_pairs:
            raise ValueError(f"duplicate client-round decision: {pair}")
        seen_pairs.add(pair)
        role = row["role"]
        expected_roles = (
            {"heterogeneous"} if client_id in h_set else
            {"malicious", "benign_designated"} if client_id in DESIGNATED_IDS else
            {"ordinary"}
        )
        if not isinstance(role, str) or role not in expected_roles:
            raise ValueError(f"inconsistent role for client {client_id}: {role!r}")
        if client_id in role_by_client and role_by_client[client_id] != role:
            raise ValueError(f"client {client_id} role changed between rounds")
        role_by_client[client_id] = role
        if type(row["accepted"]) is not bool:
            raise ValueError("accepted must be a boolean")
        weight = _finite_number(row["normalized_weight"], "normalized_weight")
        if weight < 0 or weight > 1:
            raise ValueError("normalized_weight must be between zero and one")
        if not row["accepted"] and weight != 0:
            raise ValueError("a rejected client must have exactly zero aggregation weight")
        if method == "fedavg" and not row["accepted"]:
            raise ValueError("FedAvg must accept every client")
        weights_by_round[round_id].append(weight)
        if client_id in h_set:
            h_weights.append(weight)
            h_rejected += not row["accepted"]
            h_zero_trust += weight == 0
            h_low_trust += weight < threshold
    for round_id, weights in weights_by_round.items():
        if len(weights) != CLIENT_COUNT:
            raise ValueError(f"round {round_id} does not contain every client")
        if not math.isclose(math.fsum(weights), 1.0, rel_tol=0, abs_tol=WEIGHT_SUM_ABS_TOLERANCE):
            raise ValueError(f"round {round_id} normalized weights must sum to one")
        if method == "fedavg" and any(
            not math.isclose(weight, 1 / CLIENT_COUNT, rel_tol=0,
                             abs_tol=UNIFORM_WEIGHT_ABS_TOLERANCE)
            for weight in weights
        ):
            raise ValueError(f"round {round_id} FedAvg weights must each equal 1/30 under the equal-sample protocol")

    h_count = expected_rounds * len(h_set)
    is_trust = method == "fltrust"
    adverse_count = h_low_trust if is_trust else h_rejected
    return {
        "method": canonical_names[method],
        "ordinary_accuracy": correct_counts["ordinary"] / expected_test_size,
        "heterogeneous_accuracy": correct_counts["heterogeneous"] / expected_test_size,
        "prediction_counts": prediction_counts,
        "correct_counts": correct_counts,
        "label_counts": label_counts,
        "completed_rounds": expected_rounds,
        "client_round_count": expected_count,
        "h_ids": sorted(h_set),
        "h_decision_count": h_count,
        "h_adverse_count": adverse_count,
        "h_adverse_rate": adverse_count / h_count,
        "h_adverse_metric": "low_trust_suppression_rate" if is_trust else "false_rejection_rate",
        "h_false_rejection_rate": None if is_trust else h_rejected / h_count,
        "h_low_trust_suppression_rate": h_low_trust / h_count if is_trust else None,
        "h_native_zero_trust_rate": h_zero_trust / h_count if is_trust else None,
        "low_trust_weight_threshold": threshold if is_trust else None,
        "h_relative_weight_retention": math.fsum(h_weights) / expected_rounds / (len(h_set) / CLIENT_COUNT),
        "fedavg_false_rejection_zero_by_construction": method == "fedavg",
    }
