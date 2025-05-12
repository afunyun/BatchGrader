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


import pytest
import sys

if __name__ == "__main__":
    sys.exit(pytest.main())
from datetime import datetime

test_cases = [
    {
        'name': 'Legacy Single-Batch',
        'input': 'tests/input/small_legacy.csv',
        'config': 'tests/test_config_legacy.yaml',
        'expected_output': 'tests/output/small_legacy_results.csv',
        'expected_rows': 20,
        'expected_log_msgs': ['BatchGrader run started'],
    },
    {
        'name': 'Forced Chunking',
        'input': 'tests/input/chunking_large.csv',
        'config': 'tests/test_config_forced_chunk.yaml',
        'expected_output': 'tests/output/chunking_large_forced_results.csv',
        'expected_rows': 100,
        'expected_log_msgs': ['Splitting input file'],
    },
    {
        'name': 'Token-Based Chunking',
        'input': 'tests/input/chunking_large.csv',
        'config': 'tests/test_config_token_chunk.yaml',
        'expected_output': 'tests/output/chunking_large_token_results.csv',
        'expected_rows': 100,
        'expected_log_msgs': ['Splitting input file'],
    },
    {
        'name': 'Concurrency Limit',
        'input': 'tests/input/chunking_large.csv',
        'config': 'tests/test_config_concurrency.yaml',
        'expected_output': 'tests/output/chunking_large_concurrency_results.csv',
        'expected_rows': 100,
        'expected_log_msgs': ['Splitting input file'],
    },
]
def get_latest_log():
    logs = sorted(glob.glob('output/logs/batchgrader_run_*.log'), key=os.path.getmtime, reverse=True)
    return logs[0] if logs else None
def check_output_file(path, expected_rows):
    if not os.path.exists(path):
        return False, f"Output file missing: {path}"
    try:
        df = pd.read_csv(path)
        if len(df) != expected_rows:
            return False, f"Expected {expected_rows} rows, found {len(df)}"
    except Exception as e:
        return False, f"Error reading output file: {e}"
    return True, "Output file OK"

def check_log_messages(log_path, expected_msgs):
    if not os.path.exists(log_path):
        return False, f"Log file missing: {log_path}"
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    missing = [msg for msg in expected_msgs if msg not in log_content]
    if missing:
        return False, f"Missing log messages: {missing}"
    return True, "Log messages OK"

def run_test(test):
    print(f"\n=== Running Test: {test['name']} ===")
    if 'expected_output' in test and test['expected_output']:
        try:
            if os.path.exists(test['expected_output']):
                os.remove(test['expected_output'])
        except Exception as e:
            print(f"Warning: Failed to remove old output file {test['expected_output']}: {e}")
    log_path = None
    if 'expected_log_msgs' in test:
        log_path = get_latest_log()
        try:
            if log_path and os.path.exists(log_path):
                os.remove(log_path)
        except Exception as e:
            print(f"Warning: Failed to remove old log file {log_path}: {e}")
    cmd = [
        sys.executable, '-u', 'src/batch_runner.py',
        '--config', test['config'],
        '--file', test['input']
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent.parent.resolve())
    result = subprocess.run(cmd, env=env)
    summary = {'name': test['name']}
    summary['exit_code'] = result.returncode
    if 'expected_output' in test and 'expected_rows' in test:
        ok, msg = check_output_file(test['expected_output'], test['expected_rows'])
        summary['output_file'] = msg
        summary['output_file_pass'] = ok
    if 'expected_log_msgs' in test:
        log_path = get_latest_log()
        ok, msg = check_log_messages(log_path, test['expected_log_msgs'])
        summary['log_check'] = msg
        summary['log_check_pass'] = ok
    summary['pass'] = (result.returncode == 0 and
                    summary.get('output_file_pass', True) and
                    summary.get('log_check_pass', True))
    return summary

def run_invalid_config_test(test):
    print(f"\n=== Running Invalid Config Test: {test['name']} ===")
    log_path = None
    try:
        latest_log = get_latest_log()
        if latest_log and os.path.exists(latest_log):
            os.remove(latest_log)
    except Exception as e:
        print(f"Warning: Failed to remove old log file {latest_log}: {e}")
    cmd = [
        sys.executable, '-u', 'src/batch_runner.py',
        '--config', test['config'],
        '--file', test['input']
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parent.parent.resolve())
    result = subprocess.run(cmd, env=env)
    summary = {'name': test['name']}
    summary['exit_code'] = result.returncode
    log_path = get_latest_log()
    ok, msg = check_log_messages(log_path, test['expected_log_msgs'])
    summary['log_check'] = msg
    summary['log_check_pass'] = ok
    summary['error_expected'] = test.get('expect_error', False)
    summary['pass'] = (result.returncode != 0 if test.get('expect_error', False) else result.returncode == 0) and ok
    return summary

def main():
    print("\nBatchGrader Automated Integration Test Runner\n============================================")
    results = []
    for test in test_cases:
        results.append(run_test(test))
    for test in invalid_config_cases:
        results.append(run_invalid_config_test(test))
    print("\n=== Test Summary ===")
    for r in results:
        print(f"{r['name']}: {'PASS' if r['pass'] else 'FAIL'}")
        if not r['pass']:
            print(f"  - Exit code: {r['exit_code']}")
            if 'output_file' in r:
                print(f"  - Output file: {r['output_file']}")
            if 'log_check' in r:
                print(f"  - Log check: {r['log_check']}")
    print("\nDone.")

if __name__ == "__main__":
    main()

