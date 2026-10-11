"""Deterministic workflow rehearsal; this module never trains a model.

Reference values are illustrative inputs, not per-seed observations. Generated
file hashes establish file integrity only and are not evidence of training.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any


ORIGIN = "simulated_workflow"
ATTACKS = ("none", "label_flip", "text_backdoor")
METHODS = ("FedAvg", "IndexGuard", "FLAME", "FLTrust")
METRIC_KINDS = {
    "FedAvg": "no_filtering",
    "IndexGuard": "heterogeneous_false_rejection",
    "FLAME": "heterogeneous_false_rejection",
    "FLTrust": "heterogeneous_low_trust_suppression",
}
STEPS = (
    "client_local_training", "attack", "filtering_or_weighting",
    "aggregation", "evaluation", "checkpoint",
)
NUMERIC_FIELDS = (
    "heterogeneous_misclassification_pct", "ordinary_accuracy_pct",
    "heterogeneous_accuracy_pct",
)
CSV_FIELDS = ("data_origin", "measured", "attack", "method", *NUMERIC_FIELDS,
              "metric_kind")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate(protocol: dict, reference: dict) -> list[dict]:
    seeds = protocol["seeds"]
    if seeds != [2026, 2027, 2028]:
        raise ValueError("The rehearsal scope requires seeds 2026, 2027, 2028")
    if protocol["federated"]["rounds"] != 50 or protocol["clients"]["total"] != 30:
        raise ValueError("The rehearsal scope requires 50 rounds and 30 clients")
    if not isinstance(protocol["model"]["name"], str) or not protocol["model"]["name"]:
        raise ValueError("A model name is required in the protocol")
    if (reference.get("data_origin") != "illustrative_reference"
            or reference.get("measured") is not False
            or reference.get("unit") != "percent"):
        raise ValueError("Reference provenance must identify unmeasured illustrative values")
    rows = reference.get("rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("Exactly 12 reference rows are required")
    expected = {(attack, method) for attack in ATTACKS for method in METHODS}
    seen = set()
    for row in rows:
        key = (row.get("attack"), row.get("method"))
        if key not in expected or key in seen:
            raise ValueError("Reference rows must contain each attack/method pair exactly once")
        seen.add(key)
        if row.get("metric_kind") != METRIC_KINDS[row["method"]]:
            raise ValueError(f"Incorrect metric kind for {row['method']}")
        for field in NUMERIC_FIELDS:
            value = row.get(field)
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or not 0 <= value <= 100):
                raise ValueError(f"Invalid reference percentage: {field}")
    return rows


def _events(run: dict, reference_row: dict, rounds: int) -> bytes:
    base = {
        "data_origin": ORIGIN, "measured": False, "training_executed": False,
        "run_id": run["run_id"],
    }
    records = [{**base, "event_index": 0, "event": "begin",
                "seed": run["seed"], "attack": run["attack"], "method": run["method"],
                "actual_training_status": "not_executed",
                "note": "Workflow rehearsal only; no actual training or measurement occurred.",
                "planned_rounds": rounds, "planned_clients": 30}]
    for round_number in range(1, rounds + 1):
        records.append({
            **base, "event_index": round_number, "event": "round_pipeline",
            "round": round_number,
            "steps": [{"name": name, "execution_status": "not_executed"}
                      for name in STEPS],
            "measurements": {
                "loss": None, "ordinary_accuracy": None,
                "heterogeneous_accuracy": None, "runtime_seconds": None,
                "gpu_utilization": None, "memory_bytes": None,
            },
            "client_decisions": None, "client_weights": None,
            "checkpoint_artifact": None, "checkpoint_written": False,
        })
    records.append({
        **base, "event_index": rounds + 1, "event": "end",
        "workflow_rehearsal_status": "finished",
        "reference_row": reference_row,
        "reference_row_role": "shared_illustrative_reference_not_a_seed_measurement",
        "reference_data_origin": "illustrative_reference",
        "next_step": {
            "action": "run",
            "description": "Execute this plan with the future real-training run backend.",
            "requires": ["implemented_real_training_backend", "explicit_training_authorization"],
        },
    })
    return ("\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False)
                      for item in records) + "\n").encode("utf-8")


def _existing_is_identical(output_dir: Path, files: dict[str, bytes]) -> bool:
    """Reject partial, changed, non-managed or linked output trees without writing."""
    if output_dir.is_symlink():
        raise FileExistsError("Refusing a symlink output directory")
    if not output_dir.exists():
        return False
    if not output_dir.is_dir():
        raise FileExistsError("Output path is not a directory")
    entries = list(output_dir.rglob("*"))
    if not entries:
        return False
    expected_dirs = set()
    for relative in files:
        for parent in Path(relative).parents:
            if str(parent) != ".":
                expected_dirs.add(parent.as_posix())
    actual_files = set()
    for entry in entries:
        relative = entry.relative_to(output_dir).as_posix()
        if entry.is_symlink():
            raise FileExistsError("Refusing linked output entries")
        if entry.is_dir():
            if relative not in expected_dirs:
                raise FileExistsError(f"Output contains a non-managed directory: {relative}")
        elif entry.is_file():
            actual_files.add(relative)
            if relative not in files or entry.read_bytes() != files[relative]:
                raise FileExistsError(f"Output contains non-managed or different content: {relative}")
        else:
            raise FileExistsError(f"Output contains a non-regular entry: {relative}")
    if actual_files != set(files):
        raise FileExistsError("Output contains an incomplete previous generation; refusing to overwrite")
    return True


def generate_demo(project_root: Path, output_dir: Path) -> dict:
    """Generate 36 unexecuted plans and complete 50-round rehearsal event logs.

    An existing output is accepted only if empty or byte-identical to all the
    expected generated files. Changed or additional files are never overwritten.
    """
    project_root, output_dir = Path(project_root), Path(output_dir)
    protocol_bytes = (project_root / "configs" / "protocol.json").read_bytes()
    reference_bytes = (project_root / "reference" / "table1.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    reference = json.loads(reference_bytes)
    rows = _validate(protocol, reference)
    by_key = {(row["attack"], row["method"]): row for row in rows}
    files = {"reference_table.json": reference_bytes}
    plans, index = [], []
    for seed in protocol["seeds"]:
        for attack in ATTACKS:
            for method in METHODS:
                run_id = f"seed{seed}__{attack}__{method}"
                events_path = f"runs/{run_id}/events.jsonl"
                run = {
                    "run_id": run_id, "seed": seed, "attack": attack,
                    "method": method, "status": "not_executed", "measured": False,
                    "data_origin": ORIGIN, "planned_rounds": 50,
                    "planned_clients": 30, "model_name": protocol["model"]["name"],
                    "events_path": events_path,
                    "reference_row_key": {"attack": attack, "method": method},
                }
                plans.append(run)
                files[events_path] = _events(run, by_key[(attack, method)], 50)
                index.append({"run_id": run_id, "events_path": events_path,
                              "event_count": 52, "round_pipeline_count": 50,
                              "training_status": "not_executed"})
    common = {"data_origin": ORIGIN, "measured": False, "training_executed": False}
    files["run_plan.json"] = _json_bytes({
        **common, "status": "not_executed", "run_count": 36,
        "reference_usage": "Shared illustrative values; never treated as per-seed measurements or averaged.",
        "runs": plans,
    })
    files["run_index.json"] = _json_bytes({
        **common, "runs": index, "reference_table": "reference_table.json",
        "plan": "run_plan.json", "manifest": "manifest.json",
    })
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({"data_origin": "illustrative_reference", "measured": "false",
                         **{field: row[field] for field in CSV_FIELDS[2:]}})
    files["reference_table.csv"] = csv_buffer.getvalue().encode("utf-8")
    manifest = {
        **common, "schema_version": 1,
        "integrity_scope": "SHA256 values establish integrity of generated files only; they do not establish that training occurred.",
        "manifest_self_hash_included": False,
        "inputs": [
            {"path": "configs/protocol.json", "sha256": _sha(protocol_bytes)},
            {"path": "reference/table1.json", "sha256": _sha(reference_bytes)},
        ],
        "files": [{"path": path, "bytes": len(data), "sha256": _sha(data)}
                  for path, data in sorted(files.items())],
    }
    files["manifest.json"] = _json_bytes(manifest)
    identical = _existing_is_identical(output_dir, files)
    if not identical:
        output_dir.mkdir(parents=True, exist_ok=True)
        for relative, data in sorted(files.items()):
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation also rejects unexpected files appearing after
            # the initial tree check. No existing file is ever overwritten.
            with destination.open("xb") as handle:
                handle.write(data)
    return {
        **common, "output_dir": str(output_dir.resolve()), "run_count": 36,
        "round_pipeline_event_count": 1800, "event_count": 1872,
        "reference_row_count": 12, "file_count": len(files),
        "manifest_path": str((output_dir / "manifest.json").resolve()),
        "manifest_sha256": _sha(files["manifest.json"]),
        "training_status": "not_executed",
    }
