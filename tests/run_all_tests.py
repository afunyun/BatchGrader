"""
BatchGrader Automated Integration Test Runner

Improved: Structure, safety, and maintainability. Uses typed test cases, better resource handling,
and more robust error reporting for flake-free CI and dev UX.

Usage:
    python -m tests.run_all_tests
"""

import os
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# --- Constants ---

LOGS_DIR = Path("output/logs")
PYTHON_EXEC = sys.executable

# --- Test Case Model ---


@dataclass
class TestCase:
    name: str
    input: str
    config: str
    expected_output: Optional[str] = None
    expected_rows: Optional[int] = None
    expected_log_msgs: List[str] = field(default_factory=list)
    expect_error: bool = False  # For invalid config cases


TEST_CASES: List[TestCase] = [
    TestCase(
        name="Legacy Single-Batch",
        input="tests/input/small_legacy.csv",
        config="tests/test_config_legacy.yaml",
        expected_output="tests/output/small_legacy_results.csv",
        expected_rows=20,
        expected_log_msgs=["BatchGrader run started"],
    ),
    TestCase(
        name="Forced Chunking",
        input="tests/input/chunking_large.csv",
        config="tests/test_config_forced_chunk.yaml",
        expected_output="tests/output/chunking_large_forced_results.csv",
        expected_rows=100,
        expected_log_msgs=["Splitting input file"],
    ),
    TestCase(
        name="Token-Based Chunking",
        input="tests/input/chunking_large.csv",
        config="tests/test_config_token_chunk.yaml",
        expected_output="tests/output/chunking_large_token_results.csv",
        expected_rows=100,
        expected_log_msgs=["Splitting input file"],
    ),
    TestCase(
        name="Concurrency Limit",
        input="tests/input/chunking_large.csv",
        config="tests/test_config_concurrency.yaml",
        expected_output="tests/output/chunking_large_concurrency_results.csv",
        expected_rows=100,
        expected_log_msgs=["Splitting input file"],
    ),
    TestCase(
        name="Continue on Chunk Failure",
        input="tests/input/chunk_with_failure.csv",
        config="tests/test_config_continue_on_failure.yaml",
        expected_output="tests/output/chunk_with_failure_results.csv",
        expected_rows=3,
        expected_log_msgs=[
            "Generated 5 BatchJob objects for chunk_with_failure.csv",
            "Simulating failure for chunk: chunk_with_failure_chunk_2 due to TEST_KEY_FAIL_CONTINUE",
            "Chunk chunk_with_failure_chunk_2 failed: Simulated failure for chunk_with_failure_chunk_2",
            "Simulating failure for chunk: chunk_with_failure_chunk_4 due to TEST_KEY_FAIL_CONTINUE",
            "Chunk chunk_with_failure_chunk_4 failed: Simulated failure for chunk_with_failure_chunk_4",
            "Simulating success for chunk: chunk_with_failure_chunk_1 due to TEST_KEY_FAIL_CONTINUE (non-failing chunk)",
            "Chunk chunk_with_failure_chunk_1 completed successfully.",
            "Simulating success for chunk: chunk_with_failure_chunk_3 due to TEST_KEY_FAIL_CONTINUE (non-failing chunk)",
            "Chunk chunk_with_failure_chunk_3 completed successfully.",
            "Simulating success for chunk: chunk_with_failure_chunk_5 due to TEST_KEY_FAIL_CONTINUE (non-failing chunk)",
            "Chunk chunk_with_failure_chunk_5 completed successfully.",
            "BatchGrader run finished successfully for tests/input/chunk_with_failure.csv",
        ],
    ),
]

INVALID_CONFIG_CASES: List[TestCase] = [
    TestCase(
        name="Missing Required Fields",
        input="tests/input/small_legacy.csv",
        config="tests/config/invalid_missing_required.yaml",
        expect_error=True,
        expected_log_msgs=["Missing required config fields"],
    ),
    TestCase(
        name="Invalid Output Format",
        input="tests/input/small_legacy.csv",
        config="tests/config/invalid_output_format.yaml",
        expect_error=True,
        expected_log_msgs=["Invalid output_format"],
    ),
]

# --- Utility Functions ---


def get_latest_log_file(logs_dir: Path = LOGS_DIR) -> Optional[Path]:
    """Return the most recent log file, or None if no log exists."""
    try:
        log_files = sorted(
            logs_dir.glob("batchgrader_run_*.log"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        return log_files[0] if log_files else None
    except Exception as e:
        print(f"Log file lookup error: {e}")
        return None


def remove_file_if_exists(path: Path) -> None:
    """Remove the file if it exists, suppressing errors."""
    try:
        if path.exists():
            path.unlink()
    except Exception as e:
        print(f"Warning: Failed to remove file {path}: {e}")


def read_file_content(path: Path) -> Optional[str]:
    """Read text content from a file, or return None if not readable."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not read file {path}: {e}")
        return None


def check_output_csv(path: Path, expected_rows: int) -> Tuple[bool, str]:
    """Check existence and row count of an output CSV file."""
    if not path.exists():
        return False, f"Output file missing: {path}"
    try:
        df = pd.read_csv(path)
        nrows = len(df)
        if nrows != expected_rows:
            return False, f"Expected {expected_rows} rows, found {nrows}"
    except Exception as e:
        return False, f"Error reading output file: {e}"
    return True, "Output file OK"


def check_log_messages(log_path: Path,
                       expected_msgs: List[str]) -> Tuple[bool, str]:
    """Verify that all expected log messages appear in the log file."""
    content = read_file_content(log_path)
    if content is None:
        return False, f"Log file missing: {log_path}"
    missing = [msg for msg in expected_msgs if msg not in content]
    if missing:
        return False, f"Missing log messages: {missing}"
    return True, "Log messages OK"


def run_batchgrader(test: TestCase) -> subprocess.CompletedProcess:
    """Run the BatchGrader process for a given test case."""
    cmd = [
        PYTHON_EXEC, "-u", "src/batch_runner.py", "--config", test.config,
        "--file", test.input
    ]
    # Set PYTHONPATH to project root for relative imports
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent.resolve())
    try:
        return subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
    except Exception as e:
        print(f"Error running batchgrader for test {test.name}: {e}")
        raise


def run_test_case(test: TestCase) -> Dict[str, Any]:
    """Run a single integration test and validate its output and logs."""
    print(f"\n=== Running Test: {test.name} ===")
    summary = {"name": test.name}
    output_path = Path(test.expected_output) if test.expected_output else None

    # Clean up old output file if present
    if output_path:
        remove_file_if_exists(output_path)
    # Remove latest log file before test, if relevant
    if test.expected_log_msgs:
        if latest_log := get_latest_log_file():
            remove_file_if_exists(latest_log)

    # Run process
    result = run_batchgrader(test)
    summary["exit_code"] = result.returncode
    summary["stdout"] = result.stdout
    summary["stderr"] = result.stderr

    # Output file check
    if output_path and test.expected_rows is not None:
        ok, msg = check_output_csv(output_path, test.expected_rows)
        summary["output_file"] = msg
        summary["output_file_pass"] = ok
    # Log check
    if test.expected_log_msgs:
        latest_log = get_latest_log_file()
        ok, msg = (
            False,
            "No log file") if latest_log is None else check_log_messages(
                latest_log, test.expected_log_msgs)
        summary["log_check"] = msg
        summary["log_check_pass"] = ok
    # Overall pass criteria
    summary["pass"] = (result.returncode == 0
                       and summary.get("output_file_pass", True)
                       and summary.get("log_check_pass", True))
    if not summary["pass"]:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
    return summary


def run_invalid_config_case(test: TestCase) -> Dict[str, Any]:
    """Run a negative test with invalid config and validate error logging."""
    print(f"\n=== Running Invalid Config Test: {test.name} ===")
    summary = {"name": test.name}
    # Remove latest log file before test
    latest_log = get_latest_log_file()
    if latest_log:
        remove_file_if_exists(latest_log)
    # Run process
    result = run_batchgrader(test)
    summary["exit_code"] = result.returncode
    summary["stdout"] = result.stdout
    summary["stderr"] = result.stderr

    latest_log = get_latest_log_file()
    ok, msg = (False,
               "No log file") if latest_log is None else check_log_messages(
                   latest_log, test.expected_log_msgs)
    summary["log_check"] = msg
    summary["log_check_pass"] = ok
    summary["error_expected"] = test.expect_error
    summary["pass"] = ((result.returncode != 0 if test.expect_error else
                        result.returncode == 0) and ok)
    if not summary["pass"]:
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
    return summary


def print_test_summary(results: List[Dict[str, Any]]) -> None:
    print("\n=== Test Summary ===")
    for r in results:
        status = "PASS" if r.get("pass") else "FAIL"
        print(f"{r['name']}: {status}")
        if not r.get("pass"):
            print(f"  - Exit code: {r.get('exit_code')}")
            if "output_file" in r:
                print(f"  - Output file: {r['output_file']}")
            if "log_check" in r:
                print(f"  - Log check: {r['log_check']}")
            if r.get("stdout"):
                print(f"  - STDOUT: {r['stdout'][:500]}")
            if r.get("stderr"):
                print(f"  - STDERR: {r['stderr'][:500]}")
    print("\nDone.")


def main() -> None:
    print(
        "\nBatchGrader Automated Integration Test Runner\n============================================"
    )
    results: List[Dict[str, Any]] = []
    try:
        results.extend(run_test_case(test) for test in TEST_CASES)
        results.extend(
            run_invalid_config_case(test) for test in INVALID_CONFIG_CASES)
    except Exception:
        print(f"Fatal error during test run:\n{traceback.format_exc()}")
        sys.exit(1)
    print_test_summary(results)


if __name__ == "__main__":
    main()
