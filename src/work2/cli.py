"""Commands for workflow rehearsal, prepared data, and supplied observations.

None of these commands launches a model, a GPU job or a background service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    result = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at line {line_number} of {path}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Each JSONL row must be an object: {path}:{line_number}")
        result.append(value)
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new_json(path, value):
    path = Path(path)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never replace a previous run, data split, or user-supplied result.
    with path.open("xb") as handle:
        handle.write(payload)


def analyze_supplied_run(predictions_path, decisions_path, manifest_path):
    """Compute from externally supplied records; do not certify their origin."""
    from .metrics import aggregate_run

    manifest = read_json(manifest_path)
    if manifest.get("data_origin") != "measured_training" or manifest.get("measured") is not True:
        raise ValueError("This command requires externally supplied measured-training records, not workflow examples")
    if manifest.get("status") != "completed" or manifest.get("completed_rounds") != 50:
        raise ValueError("A complete 50-round manifest is required")
    if manifest.get("seed") not in (2026, 2027, 2028):
        raise ValueError("Seed is outside this protocol")
    attack = manifest.get("attack")
    if attack not in ("none", "label_flip", "text_backdoor"):
        raise ValueError("Attack is outside this protocol")
    if manifest.get("model_name") != "Qwen/Qwen2.5-3B-Instruct":
        raise ValueError("Model does not match this protocol")
    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"]:
        raise ValueError("run_id is required")
    for field in ("dataset_manifest_sha256", "protocol_sha256"):
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"A lowercase SHA256 string is required: {field}")
    paths = {"predictions": Path(predictions_path), "decisions": Path(decisions_path)}
    receipts = {}
    for key, path in paths.items():
        digest = sha256(path)
        if manifest.get(f"{key}_sha256") != digest:
            raise ValueError(f"Manifest hash does not match {key}")
        receipts[key] = {"file_name": path.name, "sha256": digest}
    predictions, decisions = read_jsonl(predictions_path), read_jsonl(decisions_path)
    if any(row.get("measured") is False or row.get("data_origin") in
           ("illustrative_reference", "simulated_workflow") for row in predictions + decisions):
        raise ValueError("Demonstration records cannot be imported as measured observations")
    required_role = "benign_designated" if attack == "none" else "malicious"
    for decision in decisions:
        if decision.get("client_id") in range(24, 30) and decision.get("role") != required_role:
            raise ValueError("Designated attacker roles do not match the attack condition")
    metrics = aggregate_run(predictions, decisions, manifest.get("method"))
    return {
        "data_origin": "computed_from_supplied_observations",
        "input_provenance": "self_declared_by_input_producer_not_independently_verified",
        "training_executed_by_this_command": False,
        "run_id": manifest["run_id"], "seed": manifest["seed"], "attack": attack,
        "model_name": manifest["model_name"], "method": metrics["method"],
        "source_manifest_sha256": sha256(manifest_path),
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        "identity_hash_verification": "Dataset/protocol hashes are producer-declared; only predictions, decisions and supplied manifest bytes are hashed here.",
        "input_files": receipts, "metric_units": "fraction_0_to_1", "metrics": metrics,
    }


def poison_partitions(partitions, attack, seed):
    from .attacks import poison_client
    from .data import canonical_sha256, validate_partitions

    source_validation = validate_partitions(partitions)
    scheme = partitions["manifest"]["scheme"]
    required = {
        "seed": seed, "client_size": 500, "heterogeneous_count_per_H": 200,
        "ordinary_client_ids": list(range(18)),
        "heterogeneous_client_ids": list(range(18, 24)),
        "attacker_client_ids": list(range(24, 30)),
        "test_size_per_domain": 2000, "root_size": 100,
        "trigger": "The silver lantern is on the desk.",
    }
    if any(scheme.get(key) != value for key, value in required.items()):
        raise ValueError("Clean partition scheme does not match the requested seed and fixed protocol")
    clean_digest = canonical_sha256({
        "clients": {str(k): v for k, v in partitions["clients"].items()},
        "tests": partitions["tests"], "root": partitions["root"],
    })
    if partitions["manifest"].get("selected_records_sha256") != clean_digest:
        raise ValueError("Clean partition record hash does not match its manifest")
    clients, attack_manifests = {}, {}
    for client_id, records in partitions["clients"].items():
        key = int(client_id)
        clients[str(key)], attack_manifests[str(key)] = poison_client(
            records, attack, seed=seed * 1000 + key,
            designated_attacker=key in range(24, 30))
    return {
        "clients": clients, "tests": partitions["tests"], "root": partitions["root"],
        "attack": attack, "seed": seed, "training_executed": False,
        "source_partition_manifest": partitions["manifest"],
        "input_clean_partition_validation": source_validation,
        "validation_scope": "Input clean partitions only; poisoning intentionally changes labels and/or text.",
        "attack_manifests": attack_manifests,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate explicitly identified 36-run workflow examples; no model training")
    demo.add_argument("--output", type=Path, default=Path("results/workflow_demo"))
    prepare = commands.add_parser("prepare", help="Split already extracted English records by product")
    prepare.add_argument("--ordinary", type=Path, required=True)
    prepare.add_argument("--heterogeneous", type=Path, required=True)
    prepare.add_argument("--seed", type=int, choices=(2026, 2027, 2028), required=True)
    prepare.add_argument("--output", type=Path, required=True)
    poison = commands.add_parser("poison", help="Apply the attack budget to prepared client data")
    poison.add_argument("--partitions", type=Path, required=True)
    poison.add_argument("--attack", choices=("none", "label_flip", "text_backdoor"), required=True)
    poison.add_argument("--seed", type=int, choices=(2026, 2027, 2028), required=True)
    poison.add_argument("--output", type=Path, required=True)
    analyze = commands.add_parser("analyze", help="Validate and summarize externally supplied observations")
    analyze.add_argument("--predictions", type=Path, required=True)
    analyze.add_argument("--decisions", type=Path, required=True)
    analyze.add_argument("--run-manifest", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import generate_demo
            result = generate_demo(args.project_root, args.output)
        elif args.command == "prepare":
            from .data import build_partitions
            result = build_partitions(read_jsonl(args.ordinary), read_jsonl(args.heterogeneous), args.seed)
            write_new_json(args.output, result)
            result = {"status": "data_prepared", "training_executed": False, "output": str(args.output),
                      "sha256": sha256(args.output)}
        elif args.command == "poison":
            partitions = read_json(args.partitions)
            result = poison_partitions(partitions, args.attack, args.seed)
            result["source_partitions_sha256"] = sha256(args.partitions)
            write_new_json(args.output, result)
            result = {"status": "poisoned_data_prepared", "training_executed": False,
                      "output": str(args.output), "sha256": sha256(args.output)}
        else:
            result = analyze_supplied_run(args.predictions, args.decisions, args.run_manifest)
            write_new_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
