# [2025-05-11] Enforce Required Pricing CSV

- **Change:** BatchGrader now requires `docs/pricing.csv` to be present at runtime. If the file is missing, all pricing and cost estimation logic will raise a `FileNotFoundError` immediately (fail-fast behavior).
- **Rationale:** Pricing is critical for cost estimation, logging, and token tracking. Silent fallback or degraded operation can cause subtle mispricing, cost reporting errors, or downstream logic failures. By enforcing the presence of `pricing.csv`, we ensure the operator is explicitly informed and can correct the issue before running any jobs. This is a hard-stop error: the system should not proceed without pricing.
- **Impacted Modules:** `token_tracker.py`, `cost_estimator.py`, and any other code using `_load_pricing()` or cost estimation utilities.
- **Testing:** The test suite (`test_splitter.py`) explicitly checks for this error condition and now passes with the stricter logic.

---

## [2025-05-11] Deep Merge Config Enhancement

- **Change:** Configuration merging now uses a true deep merge (via `deep_merge_dicts` in `src/utils.py`). Nested dictionaries in user config files are recursively merged with `DEFAULT_CONFIG`, so user overrides only the intended keys without overwriting entire sections.
- **Rationale:** Prevents accidental loss of nested config defaults, enables more robust user customization, and aligns with best practices for configuration management in modular Python projects.
- **Implementation:**
  - Added `deep_merge_dicts` utility in `src/utils.py`.
  - Updated `load_config` in `src/config_loader.py` to use deep merge instead of shallow `dict.update()`.
- **Artifacts:**
  - Source: `src/config_loader.py`, `src/utils.py`
  - Docs: This changelog entry
- **User Impact:**
  - User `config.yaml` can now safely override only specific nested settings without breaking or losing other defaults.

---

## [2025-05-11] Version 0.3.0: Rich CLI Live Table & Summary

## [2025-05-11] Convention Restored: Chunked Inputs in input/_chunked/

- **Change:** All chunked input files are now always written to `input/_chunked/` (never cluttering the main input/ directory).
- **Rationale:**
  - Keeps user input directory clean and easy to audit.
  - Simplifies cleanup and prevents accidental data loss.
  - Makes it easy to .gitignore all chunked artifacts.
- **Implementation:**
  - Enforced in `split_file_by_token_limit` and all chunking logic in `batch_runner.py`.
  - `.gitignore` updated to exclude `input/_chunked/`.
  - Documented here and in code comments.
- **Artifacts:**
  - Source: `src/batch_runner.py`, `src/input_splitter.py`, `.gitignore`, this changelog.

- **Milestone:** Major CLI/UX upgrade for BatchGrader.
- **Features:**
  - Integrated [rich](https://rich.readthedocs.io/) live-updating job table: colorized, emoji status, in-place updates for all batch jobs.
  - Added summary table after job completion: totals for jobs, successes, failures, errors, tokens, and cost (see `print_summary_table` in `rich_display.py`).
  - Persistent logging and event file updates remain, but CLI is now much cleaner and more informative.
- **Artifacts:**
  - Source: `src/rich_display.py`, `src/batch_runner.py`, `src/llm_client.py`
  - Changelog: this entry
- **Rationale:**
  - Dramatically improves usability for multi-job runs and auditability for results/costs.
  - This completes the planned feature sprint for v0.3.0.

---

## [2025-05-11] Bugfix: Historical Token Usage Logging (Aggregate Per Batch)

- **Bug:** Historical logging for token usage and cost (output/token_usage_events.jsonl) always showed 0 after job executions.
- **Root Cause:** The function `log_token_usage_event` in `token_tracker.py` was never called after batch jobs, so the event log was never populated.
- **Fix:**
  - Added a call to `log_token_usage_event` in `process_file()` (src/batch_runner.py) after cost estimation, using aggregate input and output token counts for the batch.
  - The event is logged once per batch (not per API request), as per design decision.
  - Logging is wrapped in try/except; any logging errors are written to the persistent log.
- **Event Schema:**

  ```json
  {
    "timestamp": ISO8601,
    "api_key_prefix": str,
    "model": str,
    "input_tokens": int,
    "output_tokens": int,
    "total_tokens": int,
    "cost": float,
    "request_id": str (optional)
  }
  ```

- **Artifacts:**
  - Source: `src/batch_runner.py`, `src/token_tracker.py`
  - Log output: `output/token_usage_events.jsonl`
  - This changelog entry
- **User Impact:**
  - After each batch job, token usage and cost are now accurately logged and will show up in historical stats and summaries.

---

## [2025-05-11] Integration: Rich Live CLI Job Status Table

- **Feature:** Fully integrated the [rich](https://rich.readthedocs.io/) library into BatchGrader for a modern, live-updating CLI experience.
- **Rationale:** Eliminates console clutter from repeated print statements during batch polling, greatly improving the user experience and job traceability.
- **Implementation:**
  - Added `RichJobTable` (see `src/rich_display.py`) for live, colorized job status/progress/error table.
  - Refactored polling and job management logic in `LLMClient` and `batch_runner.py` to update job status in a single live table (not via repeated print/log).
  - Removed redundant print statements; all status/progress now flows through the live table and logger.
  - BatchJob objects (`src/batch_job.py`) track status and error_message for live updates.
  - Logging remains persistent to file for post-mortem/debug; console uses RichHandler for color and compatibility with live tables.
- **Artifacts:**
  - Source: `src/rich_display.py`, `src/batch_runner.py`, `src/llm_client.py`, `src/batch_job.py`
  - Docs: This changelog, updated README.md, version bump in pyproject.toml
- **User Impact:**
  - All job status/progress is now visible in a single, clean, updating table in the CLI.
  - No more repeated status prints; easier to monitor multiple jobs and spot failures.
  - Logging and error handling remain robust and persistent.

## [2025-05-11] Logging Reliability Improvements

- Logger now always writes a clear startup message to every log file as soon as it is instantiated, guaranteeing that all log files contain at least one entry.
- All error and early exit paths in batch_runner.py now use logger.error() before exiting, ensuring that all fatal errors are recorded in the log file, not just printed to the terminal.
- This improves traceability and debugging, especially for CI/test runs or troubleshooting failed jobs.

## [FUTURE ENHANCEMENT] Rich Live Progress/Status Display for BatchGrader (planned)

- Plan to integrate the [rich](https://rich.readthedocs.io/) library for dynamic, in-place CLI progress/status tables and colorized logs.
- Logger will use RichHandler for console output (preserving color and playing well with live tables), and FileHandler for persistent logs.
- BatchJob objects will track their current_status and error_message for live updates.
- BatchRunner will manage a list of active BatchJob objects for the live display.
- Use rich.live.Live and rich.table.Table to show a dynamically updating table of chunk/job statuses, progress bars, and errors.
- Each row: chunk name, batch_id, status (color-coded), progress bar, error info.
- Main thread runs the Live table, worker threads update job status.
- Final table update before exit for clean UX.
- File logging remains for post-mortem/debug.
- Implementation will proceed **after current test stability is confirmed** to avoid introducing instability during ongoing test runs.

## [2025-05-11] Directory Naming Standardization: input/, output/, logs/ (Singular)

- Standardized all data and log directory names to use singular form for clarity and consistency.
- **Production:**
  - `input/` for source data files
  - `output/` for results and logs
  - `output/logs/` for production log output
- **Tests:**
  - `tests/input/` for test input files
  - `tests/output/` for test output files
  - `tests/logs/` for test log output
- Renamed `tests/test_inputs/` → `tests/input/` and `tests/test_outputs/` → `tests/output/`.
- Updated all test configs and code references to match the new convention.
- Added `.keep` files to ensure log directories exist in both production and test environments.
- Updated `.gitignore` to reflect the new test output directory.
- This change eliminates confusion, improves maintainability, and aligns with modern Python project standards.

- **[2025-05-11] Bugfix:** Added `sys.path` modification at the top of `src/batch_runner.py` to ensure `from src.logger import ...` and similar imports work regardless of invocation context (CLI, subprocess, or direct run). This prevents `ModuleNotFoundError: No module named 'src'` when running tests or CLI from the project root.

## [2025-05-11] Enhancement: --config CLI Argument for Scenario-based Testing

- Added `--config` argument to `batch_runner.py` CLI. When specified, loads configuration from the given YAML file, enabling scenario-based and modular test runs.
- Enhanced `load_config` in `config_loader.py` to accept an optional config path, defaulting to `config/config.yaml`.
- Refactored config-dependent globals to be set after CLI parse and config load, ensuring correct values per run.
- This enables the test runner and manual CLI runs to use different configurations without code changes.

---

## [2025-05-11] Integration: Concurrent Processing in Main Workflow

- Integrated concurrent batch processing into the main workflow (`process_file`).
- Now, if forced chunking or token-based splitting is needed, `process_file` routes to `process_file_concurrently` for parallel execution.
- Otherwise, it uses the legacy single-batch logic for backward compatibility.
- All results and errors are saved and logged as before.
- The system is now ready for targeted testing of both single-batch and concurrent batch modes.

## [2025-05-11] Progress: Concurrent Batch Job Implementation Started

- Implementation of the concurrent batch job system has begun.
- Config layer (config_loader.py, config.yaml) and chunk job abstraction (batch_job.py) are complete.
- The `_generate_chunk_job_objects` helper is implemented and tested for both forced and token-based chunking.
- All changes are modular, backward-compatible, and thoroughly documented in this scratchpad.

## [2025-05-11] Feature: Concurrent Batch Processing & Forced Chunking

## Unified Feature Specification & Implementation Plan

### **Goal**

Enable BatchGrader to process multiple chunks of an input file concurrently via the OpenAI Batch API, and allow users to optionally force chunking of inputs for speed—all managed via config.yaml. This is a modular, backward-compatible enhancement.

---

### 1. Configuration Layer Enhancements

- **File:** `src/config_loader.py`, `config/config.yaml`
- **Actions:**
  - Add to `DEFAULT_CONFIG`:
    - `max_simultaneous_batches`: int, default 2 — Max concurrent batch jobs per input file.
    - `force_chunk_count`: int, default 0 — If >1, split input into this many chunks regardless of token limits.
    - `halt_on_chunk_failure`: bool, default True — If True, aborts remaining chunks of a file on any critical chunk failure.
  - Ensure these are loaded and validated by `load_config()`.
  - Update config file(s) and comments for clarity and discoverability.

---

### 2. BatchJob Abstraction for Chunk Management

- **File:** `src/batch_job.py` (new)
- **Class:** `BatchJob`
  - **Attributes:**
    - `chunk_data_identifier` (path to chunk's data file)
    - `chunk_df` (pandas DataFrame for chunk)
    - `system_prompt`, `response_field`, `original_source_file`, `chunk_id_str`, `llm_model`, `api_key_prefix`
    - `openai_batch_id`, `status`, `results_df`, `error_message`, `input_file_id_for_chunk`, `df_chunk_with_custom_ids`
    - `input_tokens`, `output_tokens`, `cost`
  - **Methods:**
    - `__init__`, `get_status_log_str()`
  - **Purpose:** Encapsulate state, status, and results for each chunk.

---

### 3. LLMClient Enhancements for Async-Style Operations

- **File:** `src/llm_client.py`
  - Add `submit_batch_job_async(input_file_id, source_filename_tag)` → returns OpenAI batch_job.id
  - Add `check_batch_job_status(openai_batch_id)` → retrieves OpenAI Batch object
  - Ensure `_process_batch_outputs(batch_job_obj, df_with_custom_ids)` returns `(results_df, total_input_tokens, total_output_tokens)`
  - No breaking changes to existing sync logic

---

### 4. Input Chunking Strategy Implementation

- **File:** `src/batch_runner.py`
  - Add `_generate_chunk_job_objects(...)` helper:
    - If `force_chunk_count > 1`: Split DataFrame into N chunks, save to temp files, create BatchJob for each.
    - Else: Use token-based splitting (`input_splitter.split_file_by_token_limit`), create BatchJob for each chunk file.
    - Handles empty files gracefully.
  - Output: `list[BatchJob]`

---

### 5. Concurrent Execution Orchestration

- **File:** `src/batch_runner.py`
  - Add `process_file_concurrently(...)`:
    - Uses ThreadPoolExecutor (`max_workers = max_simultaneous_batches`)
    - Maintains active futures and pending jobs
    - Submits up to N jobs at once, processes completions, halts on failure if configured
    - Aggregates results from successful chunks, includes error rows for failed chunks
  - Add `_execute_single_batch_job_task(job, llm_client, config)` worker:
    - Handles chunk loading, batch submission, polling, result processing, error handling, and token/cost logging
    - Always returns updated BatchJob object
  - Update main loop to use concurrent processing for normal batch runs

---

### 6. Token Tracking & Costing Adjustments

- **File:** `src/batch_runner.py`, `src/token_tracker.py`
  - After chunk completion, call `token_tracker.log_token_usage_event()` with per-chunk stats
  - No changes to token_tracker.py if already correct
  - Aggregate stats display remains as is

---

### 7. Robust Error Handling

- **File:** `src/batch_runner.py`
  - All worker and orchestration functions use try/except, update status and error_message on failure
  - Aggregated output always includes error rows for failed chunks
  - Halts on first critical chunk failure if `halt_on_chunk_failure` is True

---

### 8. Documentation & Testing Requirements

- **Files:** `README.md`, `docs/architecture.md`, `docs/application_flow.md`, docstrings in all new/modified code
  - Document new config options, concurrent processing, forced chunking
  - Update diagrams and flow descriptions
  - Add/expand docstrings for all new classes/functions
- **Testing:**
  - Unit tests for chunking logic
  - Integration tests for concurrent processing (various input sizes, chunk counts, parallelism, and failure scenarios)
  - Verify aggregation, error handling, and token/cost logging

---

### 9. Event Dictionary Update

| Event Name                | Payload Schema                                                | Description                                  |
|---------------------------|--------------------------------------------------------------|----------------------------------------------|
| batch_job_submitted       | {batch_id, chunk_index, total_chunks, file}                  | Emitted when a batch job is submitted        |
| batch_job_completed       | {batch_id, chunk_index, total_chunks, file, status}          | Emitted when a batch job finishes           |
| all_batches_completed     | {job_id, total_batches, results_paths, file}                 | Emitted when all batches for a file are done|

---

### Affected Files & Artifacts

- `src/config_loader.py`, `config/config.yaml` (config changes)
- `src/batch_job.py` (new class)
- `src/batch_runner.py` (orchestration, chunking, error handling)
- `src/llm_client.py` (async-style batch handling)
- `src/token_tracker.py` (token/cost logging)
- `README.md`, `docs/architecture.md`, `docs/application_flow.md` (documentation)
- Unit and integration tests
- Changelog entry in `docs/scratchpad.md`

---

## BatchGrader Architecture & Execution Flow (Function-Level)

### [2025-05-11] Feature: CLI Token/Cost Stats Reporting

- **Affected file:** `src/batch_runner.py`
- **Summary:**
  - Added `--costs` CLI flag to print token/cost usage stats (all-time, today, per-model breakdown) and exit without running a batch.
  - After any batch run, stats are now displayed automatically.
  - Uses aggregation utilities from `token_tracker.py` for reporting.
  - Output is formatted as a readable table for operator clarity.
- **Effect:**
  - Enables quick, on-demand auditing of API usage and cost without running a batch.
  - Improves transparency and post-run feedback for all users.
- **Tested:**
  - Manual validation: ran with and without `--costs`, confirmed correct stats and formatting.

---

### [2025-05-11] Feature: Enhanced Token Usage & Cost Tracking

- **Affected file:** `src/token_tracker.py`
- **Summary:**
  - Added per-request, append-only logging of every successful API call (input/output tokens, cost, timestamp, model, API key prefix) to `output/token_usage_events.jsonl`.
  - Legacy daily aggregate (`output/token_usage_log.json`) retained for API limit enforcement.
  - Added aggregation and cost computation utilities for daily, all-time, or custom window reporting.
  - Cost is computed using model pricing from `docs/pricing.csv` (per 1M tokens, input/output).
  - Only logs when a request is successful (response received).
- **Effect:**
  - Enables precise tracking of token usage and cost over time, for both compliance and analytics.
  - Supports historical audits and operator transparency.
- **Tested:**
  - Manual validation: confirmed event log appends, aggregation utilities, and cost calculations match pricing table.

### Event Dictionary (Table 1) Update

| Event Name            | Trigger                      | Payload Schema                                                                                   |
|---------------------|------------------------------|------------------------------------------------------------------------------------------------|
| token_usage_event     | On successful API response   | {timestamp, api_key_prefix, model, input_tokens, output_tokens, total_tokens, cost, request_id?} |
| token_usage_aggregate | On demand (aggregation call) | {start_date, end_date, group_by, total_tokens, total_cost, per_model_breakdown}                  |

### Rationale

- Pricing and cost are based on per 1M tokens (OpenAI standard, see docs/pricing.csv).
- Append-only JSONL ensures auditability and performance for large-scale logging.
- Aggregation utilities allow flexible reporting for ops, cost control, and compliance.
- Backwards compatibility maintained for daily API limit enforcement.

### Example Usage

- `log_token_usage_event(...)` after each successful API call.
- `get_token_usage_summary(start_date, end_date, group_by)` for analytics.
- `get_total_cost()` for billing or reporting.

---

### [2024-06-09] Refactor: Separate Overall Token Limit vs Split Token Limit

- **Affected file:** `src/batch_runner.py`
- **Summary:**
  - Refactored logic to distinguish between `token_limit` (overall batch/job cap) and `split_token_limit` (per-file split threshold).
  - `TOKEN_LIMIT` is now sourced from `token_limit` in config.yaml and governs API budget enforcement, halting, and warnings.
  - `split_token_limit` is sourced from config and used only for splitting input files.
  - Both are independently configurable and respected by the CLI.
- **Effect:**
  - Prevents conflation of limits and supports flexible, modular batch processing.
- **Tested:**
  - Manual validation with various config values for both limits.

### [2024-06-09] Bugfix: Token Limit Config Key Mismatch

- **Affected file:** `src/batch_runner.py`
- **Summary:**
  - Fixed a bug where the CLI used the wrong config key (`token_limit`) instead of the correct `split_token_limit` from `config.yaml`, causing the token split limit to default to 2,000,000 regardless of user configuration.
  - Now the CLI respects the `split_token_limit` value in config.yaml (e.g., 500,000).
- **Effect:**
  - Input splitting now correctly uses the configured token limit.
- **Tested:**
  - Manual validation with custom `split_token_limit` values; CLI now splits at the correct threshold.

### [2025-05-11] Feature: System Mode Variable (get_request_mode)

- **Affected file:** `src/batch_runner.py`
- **Summary:**
  - Added a function `get_request_mode(args)` that determines the current system operating mode for display/logging.
  - Returns "API Request/Batch mode" if running a batch job (default), or "Split/Count mode (NO REQUESTS MADE)" if either `--count-tokens` or `--split-tokens` CLI flag is used.
  - Used in the CLI banner for clear operator feedback on what the system will do.
- **Effect:**
  - Improves transparency and reduces user error/confusion by showing whether API requests will be made or not.
- **Tested:**
  - Manual validation with all CLI modes (`default`, `--count-tokens`, `--split-tokens`).

### [2024-06-09] Input Splitter: Per-File Token Count Reporting

- **Affected files:** `src/input_splitter.py`, `src/batch_runner.py`
- **Summary:**
  - Enhanced the input splitting logic to report the number of tokens in each output file after splitting, when invoked from the CLI.
  - `split_file_by_token_limit` now returns a tuple: (output_files, token_counts), where token_counts is a list of token totals for each split file.
  - The CLI prints a breakdown of token counts for each output file after splitting (e.g., `Output file: ... | Tokens: ...`).
  - The `file_split` event now includes a `token_counts` field in its payload for traceability.
- **Event Dictionary (Table 1) Update:**

  | Event Name | Payload Schema                                              | Description                  |
  | ---------- | ----------------------------------------------------------- | ---------------------------- |
  | file_split | {input_file:str, output_files:list, token_counts:list[int]} | Emitted when a file is split |

- **Tested:**
  - Manual validation with large input files and various token/row limits. Verified CLI output and event logs.

---

### [2025-05-11] Input Splitter: Token/Row Limit Config

- **Affected files:** `src/input_splitter.py`, `src/config_loader.py`, `config/config.yaml`, `config/config.yaml.example`
- **Summary:**
  - Added the ability to split input files by a configurable token and/or row limit.
  - Both `split_token_limit` (default: 500,000) and `split_row_limit` (optional, default: unlimited) are configurable in `config.yaml`.
  - If both are set, splitting occurs when either limit is reached.
  - Supported for all input formats (CSV, JSON, JSONL).
  - Limits are loaded from config via `config_loader`.
  - Emits event log lines for `input_split_config_loaded` and `file_split` (see Event Dictionary below).
  - Updated config defaults and examples for discoverability.
  - Fully documented in code and scratchpad.

- **Event Dictionary (Table 1):**

  | Event Name                | Payload Schema                      | Description                        |
  | ------------------------- | ----------------------------------- | ---------------------------------- |
  | input_split_config_loaded | {token_limit:int, row_limit:int}    | Emitted when splitter loads config |
  | file_split                | {input_file:str, output_files:list} | Emitted when a file is split       |

- **Tested:**
  - Manual validation with various token and row limits, multiple file formats.

---

### [2025-05-11] Robust Batch Halt-on-Failure Logic

- **Affected file:** `src/batch_runner.py`
- **Summary:**
  - Refactored the batch processing CLI loop to halt immediately if any input file fails (due to token cap, batch job failure, or exception).
  - `process_file()` now returns `True` on success and `False` on any error or batch failure, and the CLI loop checks this and stops further processing if a failure is detected.
  - Prevents wasteful API calls and ensures clear, user-friendly halt messages.
- **Tested:** Confirmed that after a batch failure, no further files are processed.

---

### [2025-05-11] Commented out unused prompt preparation functions

- Commented out `prepare_prompt` and `prepare_batch_prompt` in `src/evaluator.py`.
- These functions were not referenced anywhere in the codebase, and are now inert but retained for historical and documentation purposes.

---

### [2025-05-11] Planned: Input Token Counting & Splitting Utility

- New CLI options will be added to `batch_runner.py`:
  - `--count-tokens` to count tokens in input files (or a specific file)
  - `--split-tokens` to split input files into parts not exceeding the configured token limit
  - `--file <filename>` to operate on a specific file
- The splitting logic will be implemented in a new utility: `src/input_splitter.py`.
- Token counting logic will be reused from `batch_runner.py` and passed to the splitter as a callback.
- This keeps the code modular, maintainable, and testable.
- All design and implementation steps are documented here per documentation-first workflow.

### [2025-05-11] Added OpenAI Batch API 50k row cap/warning

- When processing an input file, if more than 50,000 rows are detected, the system now warns the user and caps the batch to 50,000 rows before submission to OpenAI.
- This prevents submission errors and aligns with OpenAI's documented limits.

---

### [2025-05-11] Dependency Management Policy Update

- Switched to using `requirements.txt` (managed by uv) as the canonical source of dependencies.
- `pyproject.toml` and `poetry.lock` are present for compatibility (e.g., GitHub dependency insights), but are NOT authoritative.
- All contributors should update `requirements.txt` and use `uv pip install -r requirements.txt` for installs.

---

### [2025-05-11] Accurate Batch Token Counting (system + user prompt)

- Updated `process_file` in `batch_runner.py` to count the total number of tokens submitted to the API per batch.

---

### [2025-05-11] Refactored error handling in process_file (batch_runner.py)

- On error, now writes a .log file to the output directory with only the error message (not a DataFrame with per-row errors).
- The .log file matches the input filename, but uses a .log extension. If a log file exists, a timestamp is appended.
- The new logic sums tokens for both the system prompt (from the prompt template) and the user prompt ("Please evaluate the following text: ...") for every row, matching the batch API payload structure.
- Prints total, average, and max tokens per row for all submitted tokens.
- This replaces the previous logic, which only counted tokens in the response field.
- Ensures cost estimation and batch accounting are accurate and transparent.

---

### [2025-05-10] Added file overwrite prevention

- Modified `process_file` in `batch_runner.py` to prevent overwriting existing output files
- If a file with the same name already exists in the output directory, a timestamp (YYYYMMDD_HHMMSS) is appended to the filename
- Applied the same logic to error output files
- Added appropriate logging to inform users when filenames are modified to avoid overwriting

---

### [2025-05-10] Robust Prompt Template Fallback

- If `batch_evaluation_prompt` or `batch_evaluation_prompt_generic` is missing from `prompts.yaml`, the system now falls back to the default prompt in `config_loader.py` (`DEFAULT_PROMPTS`).
- A warning is printed to stderr when a fallback is used, so the user is aware.
- This prevents crashes due to missing prompt keys and ensures batch jobs always have a valid prompt.

---

## [2025-05-10] Added LLM cost display to process_file in batch_runner.py

- Prints estimated cost after processing, using CostEstimator and token columns if present.
- Looks for columns like input_tokens/prompt_tokens and output_tokens/completion_tokens in the results DataFrame.
- Uses model name from config for cost lookup.

---

### 1. Project Folder Structure

```mermaid
BatchGrader/
├── .git/
├── .vscode/
├── batchvenv/
├── config/
│   ├── config.yaml
│   └── prompts.yaml
├── docs/
│   └── scratchpad.md
├── examples/
│   └── examples.txt
├── input/
│   └── afunyun_dataset_test.csv
├── output/
├── src/
│   ├── batch_runner.py
│   ├── config_loader.py
│   ├── data_loader.py
│   ├── evaluator.py
│   └── llm_client.py
├── requirements.txt
├── README.md
└── BATCH_API_REFERENCE.md
```

---

### 2. Module & Function Map

#### src/batch_runner.py

- **process_file(filepath)**
  - Loads config and directories.
  - Loads input data (`load_data` from `data_loader.py`).
  - Loads or builds prompt template (`load_prompt_template` from `evaluator.py`).
  - Creates LLMClient and runs batch job (`LLMClient.run_batch_job`).
- **Main Routine**
  - Checks for API key.
  - Iterates over files in input directory, calling `process_file` for each.

#### src/config_loader.py

- **ensure_config_files()**
  - Ensures config, prompts, and example files/dirs exist, creates if missing.
- **is_examples_file_default(examples_path)**
  - Checks if examples file is default or user-supplied.
- **load_config()**
  - Loads and returns configuration from YAML.

#### src/cost_estimator.py

- **estimate_cost(model, n_input_tokens, n_output_tokens)**
  - Loads batch pricing from `docs/pricing.csv` (OpenAI API per-million token rates).
  - Provides cost calculation.
  - Caches pricing for efficiency. Raises error if model not found.
  - Example usage and docstring included.
  - [Update] Removed `.replace('$','')` from CSV parsing since pricing.csv now uses plain numbers. Noted this requirement in the docstring.

#### src/data_loader.py

- **load_data(filepath)**
  - Loads data from CSV, JSON, or JSONL.
- **save_data(df, filepath)**
  - Saves DataFrame to CSV, JSON, or JSONL.

#### src/evaluator.py

- **load_prompt_template(name)**
  - Loads prompt template from config/prompts.yaml.
- **prepare_prompt(template, response)**
  - Formats a prompt with a single response.
- **prepare_batch_prompt(template, responses)**
  - Formats a batch prompt with multiple responses.

#### src/llm_client.py

- **get_default_api_key, get_default_model, ...**
  - Helpers to fetch config values.
- **class LLMClient**
  - **init**: Sets up client, loads config.
  - **_prepare_batch_requests(df, system_prompt_content, response_field_name)**
    - Prepares batch requests for OpenAI API.
  - **_upload_batch_input_file(requests_data, base_filename_for_tagging)**
    - Uploads batch file to OpenAI.
  - **_manage_batch_job(input_file_id, source_filename)**
    - Manages batch job lifecycle.
  - **run_batch_job(df, system_prompt_content, response_field_name, base_filename_for_tagging)**
    - Orchestrates the whole batch job: prepares requests, uploads, manages job, collects results.

### 5. Execution Flow (Narrative, Function-Level)

1. **Startup** (`batch_runner.py`)
   - Loads config (`load_config`)
   - Validates API key
   - Iterates over files in input directory

2. **Per File** (`process_file`)
   - Loads data (`load_data`)
   - Loads/creates prompt (`load_prompt_template`, checks if default, inserts examples)
   - Instantiates `LLMClient`
   - Calls `run_batch_job`:
     - `_prepare_batch_requests` (adds custom_id, builds requests)
     - `_upload_batch_input_file` (writes requests to temp file, uploads to OpenAI)
     - `_manage_batch_job` (creates and polls batch job)
     - `_collect_batch_results` (downloads and merges results)
   - Saves results (`save_data`)
   - On error, saves error file (with or without DataFrame, as appropriate)

3. **Supporting Modules**
   - `config_loader.py`: Handles all config, prompt, and example file logic.
   - `data_loader.py`: Handles all data I/O.
   - `evaluator.py`: Handles prompt preparation and formatting.
   - `llm_client.py`: Encapsulates all OpenAI API logic, including batch job orchestration.

---

### 6. Key Improvements Over Prior Diagram

- All actual filenames and function names are now accurate.
- LLM ops are shown as part of `LLMClient.run_batch_job`.
- Error handling and output file writing are clarified.
- The prompt preparation logic is more explicit.
- Flow is function-level, not just component-level.

---

### [FUTURE ENHANCEMENT] Rich Live Progress/Status Display for BatchGrader (planned)

- Plan to integrate the [rich](https://rich.readthedocs.io/) library for dynamic, in-place CLI progress/status tables and colorized logs.
- Logger will use RichHandler for console output (preserving color and playing well with live tables), and FileHandler for persistent logs.
- BatchJob objects will track their current_status and error_message for live updates.
- BatchRunner will manage a list of active BatchJob objects for the live display.
- Use rich.live.Live and rich.table.Table to show a dynamically updating table of chunk/job statuses, progress bars, and errors.
- Each row: chunk name, batch_id, status (color-coded), progress bar, error info.
- Main thread runs the Live table, worker threads update job status.
- Final table update before exit for clean UX.
- File logging remains for post-mortem/debug.
- Implementation will proceed **after current test stability is confirmed** to avoid introducing instability during ongoing test runs.

---

### [2025-05-11] Enforce Required Pricing CSV

- **Change:** BatchGrader now requires `docs/pricing.csv` to be present at runtime. If the file is missing, all pricing and cost estimation logic will raise a `FileNotFoundError` immediately (fail-fast behavior).
- **Rationale:** Pricing is critical for cost estimation, logging, and token tracking. Silent fallback or degraded operation can cause subtle mispricing, cost reporting errors, or downstream logic failures. By enforcing the presence of `pricing.csv`, we ensure the operator is explicitly informed and can correct the issue before running any jobs. This is a hard-stop error: the system should not proceed without pricing.
- **Impacted Modules:** `token_tracker.py`, `cost_estimator.py`, and any other code using `_load_pricing()` or cost estimation utilities.
- **Testing:** The test suite (`test_splitter.py`) explicitly checks for this error condition and now passes with the stricter logic.

---

## [2025-05-11] Deep Merge Config Enhancement

- **Change:** Configuration merging now uses a true deep merge (via `deep_merge_dicts` in `src/utils.py`). Nested dictionaries in user config files are recursively merged with `DEFAULT_CONFIG`, so user overrides only the intended keys without overwriting entire sections.
- **Rationale:** Prevents accidental loss of nested config defaults, enables more robust user customization, and aligns with best practices for configuration management in modular Python projects.
- **Implementation:**
  - Added `deep_merge_dicts` utility in `src/utils.py`.
  - Updated `load_config` in `src/config_loader.py` to use deep merge instead of shallow `dict.update()`.
- **Artifacts:**
  - Source: `src/config_loader.py`, `src/utils.py`
  - Docs: This changelog entry
- **User Impact:**
  - User `config.yaml` can now safely override only specific nested settings without breaking or losing other defaults.

---

## [2025-05-11] Version 0.3.0: Rich CLI Live Table & Summary

## [2025-05-11] Convention Restored: Chunked Inputs in input/_chunked/

- **Change:** All chunked input files are now always written to `input/_chunked/` (never cluttering the main input/ directory).
- **Rationale:**
  - Keeps user input directory clean and easy to audit.
  - Simplifies cleanup and prevents accidental data loss.
  - Makes it easy to .gitignore all chunked artifacts.
- **Implementation:**
  - Enforced in `split_file_by_token_limit` and all chunking logic in `batch_runner.py`.
  - `.gitignore` updated to exclude `input/_chunked/`.
  - Documented here and in code comments.
- **Artifacts:**
  - Source: `src/batch_runner.py`, `src/input_splitter.py`, `.gitignore`, this changelog.

- **Milestone:** Major CLI/UX upgrade for BatchGrader.
- **Features:**
  - Integrated [rich](https://rich.readthedocs.io/) live-updating job table: colorized, emoji status, in-place updates for all batch jobs.
  - Added summary table after job completion: totals for jobs, successes, failures, errors, tokens, and cost (see `print_summary_table` in `rich_display.py`).
  - Persistent logging and event file updates remain, but CLI is now much cleaner and more informative.
- **Artifacts:**
  - Source: `src/rich_display.py`, `src/batch_runner.py`, `src/llm_client.py`
  - Changelog: this entry
- **Rationale:**
  - Dramatically improves usability for multi-job runs and auditability for results/costs.
  - This completes the planned feature sprint for v0.3.0.

---

## [2025-05-11] Bugfix: Historical Token Usage Logging (Aggregate Per Batch)

- **Bug:** Historical logging for token usage and cost (output/token_usage_events.jsonl) always showed 0 after job executions.
- **Root Cause:** The function `log_token_usage_event` in `token_tracker.py` was never called after batch jobs, so the event log was never populated.
- **Fix:**
  - Added a call to `log_token_usage_event` in `process_file()` (src/batch_runner.py) after cost estimation, using aggregate input and output token counts for the batch.
  - The event is logged once per batch (not per API request), as per design decision.
  - Logging is wrapped in try/except; any logging errors are written to the persistent log.
- **Event Schema:**

  ```json
  {
    "timestamp": ISO8601,
    "api_key_prefix": str,
    "model": str,
    "input_tokens": int,
    "output_tokens": int,
    "total_tokens": int,
    "cost": float,
    "request_id": str (optional)
  }
  ```

- **Artifacts:**
  - Source: `src/batch_runner.py`, `src/token_tracker.py`
  - Log output: `output/token_usage_events.jsonl`
  - This changelog entry
- **User Impact:**
  - After each batch job, token usage and cost are now accurately logged and will show up in historical stats and summaries.

---

## [2025-05-11] Integration: Rich Live CLI Job Status Table

- **Feature:** Fully integrated the [rich](https://rich.readthedocs.io/) library into BatchGrader for a modern, live-updating CLI experience.
- **Rationale:** Eliminates console clutter from repeated print statements during batch polling, greatly improving the user experience and job traceability.
- **Implementation:**
  - Added `RichJobTable` (see `src/rich_display.py`) for live, colorized job status/progress/error table.
  - Refactored polling and job management logic in `LLMClient` and `batch_runner.py` to update job status in a single live table (not via repeated print/log).
  - Removed redundant print statements; all status/progress now flows through the live table and logger.
  - BatchJob objects (`src/batch_job.py`) track status and error_message for live updates.
  - Logging remains persistent to file for post-mortem/debug; console uses RichHandler for color and compatibility with live tables.
- **Artifacts:**
  - Source: `src/rich_display.py`, `src/batch_runner.py`, `src/llm_client.py`, `src/batch_job.py`
  - Docs: This changelog, updated README.md, version bump in pyproject.toml
- **User Impact:**
  - All job status/progress is now visible in a single, clean, updating table in the CLI.
  - No more repeated status prints; easier to monitor multiple jobs and spot failures.
  - Logging and error handling remain robust and persistent.

## [2025-05-11] Logging Reliability Improvements

- Logger now always writes a clear startup message to every log file as soon as it is instantiated, guaranteeing that all log files contain at least one entry.
- All error and early exit paths in batch_runner.py now use logger.error() before exiting, ensuring that all fatal errors are recorded in the log file, not just printed to the terminal.
- This improves traceability and debugging, especially for CI/test runs or troubleshooting failed jobs.

---

### [2025-05-11] Chunked Input Directory Management & Pruning

- All chunked input files are now written to a dedicated `_chunked/` subdirectory under the relevant input directory (e.g., `input/_chunked/`, `tests/input/_chunked/`).
- At the end of each batch job (success or failure), all files in the `_chunked/` directory are deleted except for `.keep` (which is maintained for version control and test hygiene).
- This ensures that only original user-provided inputs persist, avoids clutter and disk bloat, and guarantees reproducibility for test runs.
- Implementation: Updated `split_file_by_token_limit` to write chunks to `_chunked/` and always create a `.keep` file. Added `prune_chunked_dir` utility and integrated it into the batch runner after chunk processing.
- **Event Dictionary Entry:**
  - `chunked_prune`: `{ 'chunked_dir': str, 'files_deleted': list }` — Emitted when chunked input files are pruned after batch job completion.
- Rationale: Keeps input directories clean, prevents test artifacts from accumulating, and ensures that only user data is preserved. This is especially important for CI/test environments and reproducible research.

---

### Last updated: 2025-05-11

### UPDATE INSTRUCTIONS

**Purpose:**
Update the project's version and changelog after making changes. This includes modifying `pyproject.toml`, updating `README.md`, and (optionally) fetching the latest API pricing data from OpenAI.

---

### 1. Determine the Significance of Changes

Analyze recent commits and codebase changes to decide the version increment, following semantic versioning:

- **Patch Version (`0.0.X`)**: For minor changes (bug fixes, small tweaks). Increment by `0.0.1`.
- **Minor Version (`0.X.0`)**: For new features or significant refactors that are backward-compatible. Increment by `0.1.0`.
- **Major Version (`X.0.0`)**: For substantial, backward-incompatible changes (e.g., API changes, major restructuring). Increment by `1.0.0`.

---

### 2. Update `pyproject.toml`

1. Open `pyproject.toml`.
2. Locate the `version` field under `[tool.poetry]` (or `[project]` for other build systems).
3. Update the version string according to the determination made in step 1.

**Examples:**

- Minor update:
  - Old: `version = "0.2.5"`
  - New: `version = "0.3.0"`
- Patch update:
  - Old: `version = "0.2.5"`
  - New: `version = "0.2.6"`

---

### 3. Update `README.md`

1. Find the section starting with `## Last updated:`.
2. Update the `<DATE>` placeholder in this header to the current date (`YYYY-MM-DD`).
3. Under this updated header, add a new entry for the version increase. This entry should briefly summarize the changes corresponding to the version bump. New entries should be placed above older ones.

**Example:**

If the `README.md` previously contained:

### Last updated: 2025-05-10

### Version 0.2.5

- Fixed a data parsing error.

And you are updating to version 0.3.0 on May 11, 2025, with new functionality:
Markdown

### Last updated: 2025-05-11

### Version 0.3.0

- Added a new module for advanced analytics.
- Refactored the core authentication library.

### Version 0.2.5

- Fixed a data parsing error.

```markdown

Follow these instructions to make the following change to my code document.

Instruction: Add a section capturing the design plan for integrating rich and live progress/status display into BatchGrader, just below the current top entry. Clearly mark it as a future enhancement and reference that it will be implemented after current test stability is confirmed.

Code Edit:
```

## [2025-05-11] Enforce Required Pricing CSV

- **Change:** BatchGrader now requires `docs/pricing.csv` to be present at runtime. If the file is missing, all pricing and cost estimation logic will raise a `FileNotFoundError` immediately (fail-fast behavior).
- **Rationale:** Pricing is critical for cost estimation, logging, and token tracking. Silent fallback or degraded operation can cause subtle mispricing, cost reporting errors, or downstream logic failures. By enforcing the presence of `pricing.csv`, we ensure the operator is explicitly informed and can correct the issue before running any jobs. This is a hard-stop error: the system should not proceed without pricing.
- **Impacted Modules:** `token_tracker.py`, `cost_estimator.py`, and any other code using `_load_pricing()` or cost estimation utilities.
- **Testing:** The test suite (`test_splitter.py`) explicitly checks for this error condition and now passes with the stricter logic.

---

## [2025-05-11] Deep Merge Config Enhancement

- **Change:** Configuration merging now uses a true deep merge (via `deep_merge_dicts` in `src/utils.py`). Nested dictionaries in user config files are recursively merged with `DEFAULT_CONFIG`, so user overrides only the intended keys without overwriting entire sections.
- **Rationale:** Prevents accidental loss of nested config defaults, enables more robust user customization, and aligns with best practices for configuration management in modular Python projects.
- **Implementation:**
  - Added `deep_merge_dicts` utility in `src/utils.py`.
  - Updated `load_config` in `src/config_loader.py` to use deep merge instead of shallow `dict.update()`.
- **Artifacts:**
  - Source: `src/config_loader.py`, `src/utils.py`
  - Docs: This changelog entry
- **User Impact:**
  - User `config.yaml` can now safely override only specific nested settings without breaking or losing other defaults.

---

## [2025-05-11] Version 0.3.0: Rich CLI Live Table & Summary

## [2025-05-11] Convention Restored: Chunked Inputs in input/_chunked/

- **Change:** All chunked input files are now always written to `input/_chunked/` (never cluttering the main input/ directory).
- **Rationale:**
  - Keeps user input directory clean and easy to audit.
  - Simplifies cleanup and prevents accidental data loss.
  - Makes it easy to .gitignore all chunked artifacts.
- **Implementation:**
  - Enforced in `split_file_by_token_limit` and all chunking logic in `batch_runner.py`.
  - `.gitignore` updated to exclude `input/_chunked/`.
  - Documented here and in code comments.
- **Artifacts:**
  - Source: `src/batch_runner.py`, `src/input_splitter.py`, `.gitignore`, this changelog.

- **Milestone:** Major CLI/UX upgrade for BatchGrader.
- **Features:**
  - Integrated [rich](https://rich.readthedocs.io/) live-updating job table: colorized, emoji status, in-place updates for all batch jobs.
  - Added summary table after job completion: totals for jobs, successes, failures, errors, tokens, and cost (see `print_summary_table` in `rich_display.py`).
  - Persistent logging and event file updates remain, but CLI is now much cleaner and more informative.
- **Artifacts:**
  - Source: `src/rich_display.py`, `src/batch_runner.py`, `src/llm_client.py`
  - Changelog: this entry
- **Rationale:**
  - Dramatically improves usability for multi-job runs and auditability for results/costs.
  - This completes the planned feature sprint for v0.3.0.

---

## [2025-05-11] Bugfix: Historical Token Usage Logging (Aggregate Per Batch)

- **Bug:** Historical logging for token usage and cost (output/token_usage_events.jsonl) always showed 0 after job executions.
- **Root Cause:** The function `log_token_usage_event` in `token_tracker.py` was never called after batch jobs, so the event log was never populated.
- **Fix:**
  - Added a call to `log_token_usage_event` in `process_file()` (src/batch_runner.py) after cost estimation, using aggregate input and output token counts for the batch.
  - The event is logged once per batch (not per API request), as per design decision.
  - Logging is wrapped in try/except; any logging errors are written to the persistent log.
- **Event Schema:**

  ```json
  {
    "timestamp": ISO8601,
    "api_key_prefix": str,
    "model": str,
    "input_tokens": int,
    "output_tokens": int,
    "total_tokens": int,
    "cost": float,
    "request_id": str (optional)
  }
  ```

- **Artifacts:**
  - Source: `src/batch_runner.py`, `src/token_tracker.py`
  - Log output: `output/token_usage_events.jsonl`
  - This changelog entry
- **User Impact:**
  - After each batch job, token usage and cost are now accurately logged and will show up in historical stats and summaries.

---

## [2025-05-11] Integration: Rich Live CLI Job Status Table

- **Feature:** Fully integrated the [rich](https://rich.readthedocs.io/) library into BatchGrader for a modern, live-updating CLI experience.
- **Rationale:** Eliminates console clutter from repeated print statements during batch polling, greatly improving the user experience and job traceability.
- **Implementation:**
  - Added `RichJobTable` (see `src/rich_display.py`) for live, colorized job status/progress/error table.
  - Refactored polling and job management logic in `LLMClient` and `batch_runner.py` to update job status in a single live table (not via repeated print/log).
  - Removed redundant print statements; all status/progress now flows through the live table and logger.
  - BatchJob objects (`src/batch_job.py`) track status and error_message for live updates.
  - Logging remains persistent to file for post-mortem/debug; console uses RichHandler for color and compatibility with live tables.
- **Artifacts:**
  - Source: `src/rich_display.py`, `src/batch_runner.py`, `src/llm_client.py`, `src/batch_job.py`
  - Docs: This changelog, updated README.md, version bump in pyproject.toml
- **User Impact:**
  - All job status/progress is now visible in a single, clean, updating table in the CLI.
  - No more repeated status prints; easier to monitor multiple jobs and spot failures.
  - Logging and error handling remain robust and persistent.

## [2025-05-11] Logging Reliability Improvements

- Logger now always writes a clear startup message to every log file as soon as it is instantiated, guaranteeing that all log files contain at least one entry.
- All error and early exit paths in batch_runner.py now use logger.error() before exiting, ensuring that all fatal errors are recorded in the log file, not just printed to the terminal.
- This improves traceability and debugging, especially for CI/test runs or troubleshooting failed jobs.

## [FUTURE ENHANCEMENT] Rich Live Progress/Status Display for BatchGrader (planned)

- Plan to integrate the [rich](https://rich.readthedocs.io/) library for dynamic, in-place CLI progress/status tables and colorized logs.
- Logger will use RichHandler for console output (preserving color and playing well with live tables), and FileHandler for persistent logs.
- BatchJob objects will track their current_status and error_message for live updates.
- BatchRunner will manage a list of active BatchJob objects for the live display.
- Use rich.live.Live and rich.table.Table to show a dynamically updating table of chunk/job statuses, progress bars, and errors.
- Each row: chunk name, batch_id, status (color-coded), progress bar, error info.
- Main thread runs the Live table, worker threads update job status.
- Final table update before exit for clean UX.
- File logging remains for post-mortem/debug.
- Implementation will proceed **after current test stability is confirmed** to avoid introducing instability during ongoing test runs.

{{ ... }}
