"""Write a receipt for actual local software tests, not training evidence."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import re
import subprocess
import sys

from check_workflow import check


def main():
    root = Path(__file__).resolve().parents[1]
    result_dir = root / "results"
    log_path = result_dir / "software_tests.log"
    receipt_path = result_dir / "software_checks.json"
    if log_path.exists() or receipt_path.exists():
        raise FileExistsError("Keep previous software receipts; use a new checkout for a new recorded check.")
    started = datetime.now(timezone.utc).isoformat()
    test_run = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                              cwd=root, capture_output=True, text=True)
    output = test_run.stdout + test_run.stderr
    print(output, end="")
    integrity = check(root)
    result_dir.mkdir(parents=True, exist_ok=True)
    header = "Actual software unit/integration checks. No Qwen training or experimental performance measurement.\n"
    log_bytes = (header + output).encode("utf-8")
    log_path.write_bytes(log_bytes)
    match = re.search(r"Ran (\d+) tests? in", output)
    source_paths = sorted(
        [*root.glob("src/**/*.py"), *root.glob("tests/**/*.py"), *root.glob("scripts/**/*.py"),
         root / "run.py", root / "configs/protocol.json", root / "reference/table1.json", root / "commands.json"]
    )
    receipt = {
        "record_type": "executed_software_checks",
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "command": "python -m unittest discover -s tests -v",
        "exit_code": test_run.returncode,
        "tests_run": int(match.group(1)) if match else None,
        "tests_passed": test_run.returncode == 0 and match is not None,
        "model_training_executed": False,
        "scope": "Checks demonstrate code behavior and generated-file integrity only, not Table 1 empirical validity.",
        "workflow_integrity": integrity,
        "test_log_sha256": hashlib.sha256(log_bytes).hexdigest(),
        "source_files": [{"path": str(path.relative_to(root)).replace("\\", "/"),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in source_paths],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return test_run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
