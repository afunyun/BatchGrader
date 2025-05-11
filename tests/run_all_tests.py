"""
BatchGrader Test Runner Helper

Runs a suite of manual/automated tests for BatchGrader concurrent batch processing.
Each test uses a specific config and input, then prints/logs results for manual validation.

Usage:
    python tests/run_all_tests.py
"""
import subprocess
import sys
from pathlib import Path

test_cases = [
    {
        'name': 'Legacy Single-Batch',
        'input': 'tests/input/small_legacy.csv',
        'config': 'tests/test_config_legacy.yaml',
    },
    {
        'name': 'Forced Chunking',
        'input': 'tests/input/chunking_large.csv',
        'config': 'tests/test_config_forced_chunk.yaml',
    },
    {
        'name': 'Token-Based Chunking',
        'input': 'tests/input/chunking_large.csv',
        'config': 'tests/test_config_token_chunk.yaml',
    },
    {
        'name': 'Concurrency Limit',
        'input': 'tests/input/chunking_large.csv',
        'config': 'tests/test_config_concurrency.yaml',
    },
    {
        'name': 'Halt on Chunk Failure',
        'input': 'tests/input/corrrupt.csv',
        'config': 'tests/test_config_halt_on_failure.yaml',
    },
    {
        'name': 'Empty Input',
        'input': 'tests/input/empty.csv',
        'config': 'tests/test_config_legacy.yaml',
    },
]

def run_test(test):
    print(f"\n=== Running Test: {test['name']} ===")
    cmd = [
        sys.executable, '-u', 'src/batch_runner.py',
        '--config', test['config'],
        '--file', test['input'],
        '--log_dir', 'tests/logs'
    ]
    try:
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print(f"[PASS] {test['name']} completed.")
        else:
            print(f"[FAIL] {test['name']} failed (exit code {result.returncode}).")
    except Exception as e:
        print(f"[ERROR] Could not run {test['name']}: {e}")

def main():
    print("\nBatchGrader Test Runner\n========================")
    for idx, test in enumerate(test_cases, 1):
        print(f"{idx}. {test['name']}")
    print("A. Run ALL tests\n")
    choice = input("Select a test to run (number or A for all): ").strip().lower()
    if choice in ("a", "all", ""):  # Run all if blank or 'a'
        for test in test_cases:
            run_test(test)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(test_cases):
                run_test(test_cases[idx])
            else:
                print("Invalid selection. Exiting.")
        except Exception:
            print("Invalid input. Exiting.")

if __name__ == "__main__":
    main()

