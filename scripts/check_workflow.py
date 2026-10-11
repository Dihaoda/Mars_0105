"""Verify generated-file integrity and unmeasured workflow provenance."""
from pathlib import Path
import hashlib
import json
import sys


def check(root):
    root = Path(root).resolve()
    demo = root / "results" / "workflow_demo"
    manifest = json.loads((demo / "manifest.json").read_text(encoding="utf-8"))
    if manifest["data_origin"] != "simulated_workflow" or manifest["measured"] is not False:
        raise ValueError("Invalid demonstration provenance")
    for base, entries in ((root, manifest["inputs"]), (demo, manifest["files"])):
        for entry in entries:
            target = (base / entry["path"]).resolve()
            if not target.is_relative_to(base):
                raise ValueError("Manifest path escapes its root")
            data = target.read_bytes()
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"Hash mismatch: {entry['path']}")
            if "bytes" in entry and len(data) != entry["bytes"]:
                raise ValueError(f"Length mismatch: {entry['path']}")
    reference = json.loads((root / "reference" / "table1.json").read_text(encoding="utf-8"))
    reference_map = {(r["attack"], r["method"]): r for r in reference["rows"]}
    plan = json.loads((demo / "run_plan.json").read_text(encoding="utf-8"))
    if len(plan["runs"]) != 36:
        raise ValueError("Expected 36 planned cases")
    seen, round_count, event_count = set(), 0, 0
    for run in plan["runs"]:
        key = (run["seed"], run["attack"], run["method"])
        if key in seen or run["status"] != "not_executed":
            raise ValueError("Duplicate or incorrectly labelled plan")
        seen.add(key)
        rows = [json.loads(line) for line in (demo / run["events_path"]).read_text(encoding="utf-8").splitlines()]
        if len(rows) != 52 or rows[0]["event"] != "begin" or rows[-1]["event"] != "end":
            raise ValueError("Incomplete workflow example")
        for index, row in enumerate(rows):
            if (row["data_origin"] != "simulated_workflow" or row["measured"] is not False
                or row["training_executed"] is not False or row["event_index"] != index):
                raise ValueError("Event provenance/order mismatch")
        for number, row in enumerate(rows[1:-1], 1):
            if row["event"] != "round_pipeline" or row["round"] != number:
                raise ValueError("Round sequence mismatch")
            if any(value is not None for value in row["measurements"].values()):
                raise ValueError("Unmeasured examples must not contain measured telemetry")
            if row["checkpoint_written"] is not False or row["checkpoint_artifact"] is not None:
                raise ValueError("A workflow example cannot claim a model checkpoint")
        if rows[-1]["reference_row"] != reference_map[(run["attack"], run["method"])]:
            raise ValueError("Reference table changed in the example")
        round_count += 50
        event_count += len(rows)
    return {
        "check_type": "generated_workflow_integrity_and_provenance",
        "checks_passed": True, "file_count": len(manifest["files"]),
        "run_plan_count": len(seen), "round_event_count": round_count,
        "event_count": event_count, "reference_rows": len(reference_map),
        "model_training_executed": False,
        "manifest_sha256": hashlib.sha256((demo / "manifest.json").read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    print(json.dumps(check(root), indent=2))
