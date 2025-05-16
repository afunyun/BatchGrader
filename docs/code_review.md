CODE REVIEW FOR src/batch_runner.py AND RELATED MODULES

- **Duplicated Code:**
  - **Status:** Addressed.
  - **Description:**
    - Token counting logic moved to `token_utils.py`.
    - LLM client initialization and token logging centralized in `llm_utils.py` (though `llm_utils.py` itself was later integrated or its functions moved).
    - Similar file processing logic in `process_file` and `process_file_concurrently` now consolidated.
  - **Recommendation:**
    1. Consolidate LLM client management.
    2. Refactor file processing to reduce duplication.
    3. Move token limit checking to `token_utils`.
  - **Changes Made:**
    - Utility functions in `token_utils.py` created and refined.
    - `file_processor.py` created and significantly enhanced:
        - `process_file_wrapper` provides a unified entry point for processing individual files.
        - `process_file_common` handles shared logic like data loading, initial token checks, and deciding between sequential/concurrent processing.
        - All concurrent processing logic (`process_file_concurrently` and its helpers like `_generate_chunk_job_objects`, `_execute_single_batch_job_task`, etc.) moved from `batch_runner.py` to `file_processor.py`, resolving circular dependencies and centralizing this complex logic.
    - `batch_runner.process_file` refactored to be a lightweight wrapper around `file_processor.process_file_wrapper`.

- **Broad Exception Handling:**
  - **Status:** Addressed. Specific exceptions are now caught more frequently, and a general fallback includes a full stack trace where appropriate.
  - **Description:** Many try-except blocks catch all exceptions without specificity (e.g., lines 641-650, lines 679-688), which can mask critical errors.
  - **Recommendation:** Catch specific exceptions (such as `FileNotFoundError`, `ValueError`) and handle them appropriately. Log unexpected exceptions with detailed stack traces, but avoid catching all exceptions unless absolutely necessary.
  - **Changes Made:**
    - Added `DEFAULT_GLOBAL_TOKEN_LIMIT` and `DEFAULT_SPLIT_TOKEN_LIMIT` to `src/constants.py`.
    - Added `DEFAULT_RESPONSE_FIELD` to `src/constants.py`.
    - Added `DEFAULT_LOG_DIR` and `DEFAULT_ARCHIVE_DIR` to `src/constants.py`, with `LOG_DIR` and `ARCHIVE_DIR` being mutable for CLI overrides (handled in `batch_runner.run_batch_processing` and `cli.py`).
    - `batch_runner.py` and `file_processor.py` updated to import and use these constants.
    - The `"custom_id"` string is generally handled by `LLMClient` or `BatchJob`.

- **Verbose Logging:**
  - **Status:** Addressed. Logging for tiktoken issues has been consolidated.
  - **Description:** Logging is overly verbose with redundant messages (e.g., multiple error logs for tiktoken issues on lines 607-613).
  - **Recommendation:** Consolidate related log messages and use appropriate log levels (`debug` for details, `info` for key steps, `error` for failures) to reduce noise.

- **Hardcoded Values and Magic Strings/Numbers:**
  - **Status:** Addressed.
  - **Description:** Hardcoded values like `TOKEN_LIMIT = 2_000_000` and `split_token_limit = 500_000` were present in `batch_runner.py`. Magic strings like `"custom_id"` were also noted.
  - **Recommendation:** Move these to a configuration file or a dedicated constants module (`src/constants.py`) to improve maintainability and clarity.
  - **Changes Made:**
    - Added `DEFAULT_GLOBAL_TOKEN_LIMIT` and `DEFAULT_SPLIT_TOKEN_LIMIT` to `src/constants.py`.
    - Added `DEFAULT_RESPONSE_FIELD` to `src/constants.py`.
    - Added `DEFAULT_LOG_DIR` and `DEFAULT_ARCHIVE_DIR` to `src/constants.py`, with `LOG_DIR` and `ARCHIVE_DIR` being mutable for CLI overrides.
    - `batch_runner.py` and `file_processor.py` will be updated to import these constants.
    - The `"custom_id"` string is often related to DataFrame column names and is generally acceptable if consistently used and named, but direct LLM client interaction specifics (like custom_id expectations) are better managed within the `LLMClient` or `BatchJob` classes.

- **Mixed Responsibilities in Functions:**
  - **Status:** Addressed.
  - **Description:** Functions like `process_file` in `batch_runner.py` and concurrent helpers were doing too much.
  - **Recommendation:** Break into smaller functions. Move Rich-table UI updates to a distinct layer.
  - **Changes Made:**
    - `file_processor.py` now has a clearer separation of concerns for file processing (sequential, concurrent, common utilities).
    - Concurrent processing helpers (`_generate_chunk_job_objects`, `_execute_single_batch_job_task`, `_pfc_submit_jobs`, `_pfc_process_completed_future`, `_pfc_aggregate_and_cleanup`) are now private to `file_processor.py`.
    - `batch_runner.process_file` simplified.
    - CLI logic moved to `src/cli.py`.

- **Inconsistent File Path Handling:**
  - **Status:** Addressed.
  - **Description:** Mixed `os.path` and `pathlib.Path`.
  - **Recommendation:** Standardize on `pathlib.Path`.
  - **Changes Made:** Code across `batch_runner.py`, `file_processor.py`, and the new `cli.py` has been updated to use `pathlib.Path` more consistently.

- **CLI Argument Handling Mixed with Logic:**
  - **Status:** Addressed.
  - **Description:** CLI parsing was in `batch_runner.py`'s `__main__`.
  - **Recommendation:** Move CLI parsing to `src/cli.py`.
  - **Changes Made:** 
    - `src/cli.py` created, containing all `argparse` logic in its `main()` function.
    - `cli.main()` now parses arguments, loads config, and calls `batch_runner.run_batch_processing(args, config)`.
    - `batch_runner.py`'s `__main__` block now calls `cli.main()`.
    - `batch_runner.run_batch_processing` handles application logic post-parsing (e.g., setting `LOG_DIR` from args).

- **Unused or Redundant Code:**
  - **Status:** Addressed.
  - **Description:** Placeholder `finally` in `process_file_concurrently` (when in batch_runner), duplicate `ArgumentParser`.
  - **Recommendation:** Remove unused code and duplicate parser.
  - **Changes Made:**
    - Redundant concurrent functions removed from `batch_runner.py`.
    - `sys.path.insert` removed from `batch_runner.py`.
    - `batch_runner.main()` (with mock args) removed.
    - Duplicate parser implicitly removed by centralizing CLI logic in `cli.py`.

- **Code Length and Complexity:**
  - **Status:** Addressed.
  - **Description:** `batch_runner.py` was too long and complex.
  - **Recommendation:** Split into smaller, focused modules.
  - **Changes Made:** 
    - `batch_runner.py` significantly slimmed down by moving concurrent logic to `file_processor.py` and CLI logic to `cli.py`.
    - `file_processor.py` now handles the complexity of file processing, but is itself focused on that domain.
    - `cli.py` handles command-line interaction.
    - **NEW**: `batch_runner.py` now includes `run_count_mode` and `run_split_mode` functions, callable via `cli.py` using `--mode count` or `--mode split`. These modes allow users to specifically count tokens in files or split files based on token/row limits without full batch processing.

- **Circular Imports:**
  - **Status:** Addressed.
  - **Description:** Potential circular dependency between `batch_runner.py` and `file_processor.py` if `process_file_concurrently` called back into `batch_runner` or vice-versa inappropriately.
  - **Recommendation:** Ensure clear separation and unidirectional dependencies where possible, or careful structuring of calls.
  - **Changes Made:** Moving `process_file_concurrently` and its helpers entirely into `file_processor.py` resolved the primary circular dependency risk associated with these functions. `batch_runner` now calls `file_processor` for these operations.

- **Configuration Loading and Management:**
  - **Status:** Improved.
  - **Description:** Config loading was somewhat spread.
  - **Recommendation:** Centralize config loading and access.
  - **Changes Made:** `config_loader.py` handles loading. `cli.py` loads config and passes it down. `batch_runner.py` and `file_processor.py` receive and use the config dict.

- **Logging Initialization and Configuration:**
    - **Status:** Improved.
    - **Description:** `prune_logs_if_needed` was called at the top level of `batch_runner.py` before CLI `--log_dir` could be processed.
    - **Recommendation:** Ensure logger and its dependent functions (like pruning) are initialized/called after CLI arguments can modify paths.
    - **Changes Made:** The `prune_logs_if_needed` call was moved into `batch_runner.run_batch_processing` after `LOG_DIR` and `ARCHIVE_DIR` (global constants from `src.constants`) are potentially updated by the `args.log_dir` value passed from `cli.py`. For `run_count_mode` and `run_split_mode`, log pruning is currently skipped as these modes are generally non-invasive to the log directory structure for batch results.

- **Unit Test Coverage & Maintenance:**
  - **Status:** Improved / In Progress.
  - **Description:** Original unit tests were primarily in `tests/test_batch_runner.py` and focused on concurrent processing. Other modules like `llm_utils.py` had some tests.
  - **Recommendation:** Review and update unit tests to align with the refactored code structure. Ensure new logic in `file_processor.py`, `cli.py`, and the modified `batch_runner.py` is covered. Create new test files as needed (e.g., `test_cli.py`, `test_file_processor.py`).
  - **Changes Made:**
    - Tests for concurrent file processing (e.g., `test_continue_on_chunk_failure`) moved from `tests/test_batch_runner.py` to `tests/test_file_processor.py` and adapted.
    - `tests/test_file_processor.py` enhanced with more robust testing for chunk failure scenarios (simulating full chunk failure with `halt_on_chunk_failure: false`).
    - `tests/test_batch_runner.py` cleaned up; new focused tests added for its remaining responsibilities (`run_batch_processing`, `process_file`, `run_count_mode`, `run_split_mode`).
    - New test file `tests/test_cli.py` created with comprehensive tests for command-line argument parsing and dispatch logic in `src/cli.py`.
    - Fixtures in test files were reviewed and adapted for the new structure.

- **Error Handling & Logging in Concurrent Processing:**
  - **Status:** Improved.
  - **Description:** Error handling and logging in concurrent processing was not consistently handled.
  - **Recommendation:** Ensure errors are logged and handled appropriately in concurrent processing.
  - **Changes Made:**
    - `file_processor.py` now logs errors and handles exceptions in concurrent processing more consistently.
    - `batch_runner.py`'s `run_batch_processing` function now catches and logs exceptions from concurrent processing.

**New/Enhanced Features:**
- **Operational Modes (`--mode`):
  - **Status:** Implemented.
  - **Description:** The CLI now supports `--mode batch` (default), `--mode count`, and `--mode split`.
    - `batch`: Performs the standard batch processing of files with LLMs.
    - `count`: Iterates through input file(s), loads data, and reports token statistics (total, average, max per row) for each file based on the configured system prompt and response field. Uses `file_processor.check_token_limits` with a high token threshold to gather stats.
    - `split`: Iterates through input file(s) and splits them into smaller chunks based on `split_token_limit` (or `input_splitter_options.max_tokens_per_chunk`), `input_splitter_options.max_rows_per_chunk`, and `input_splitter_options.force_chunk_count` from the configuration. Uses `input_splitter.split_file_by_token_limit`. Generated chunks are saved in a `_chunked` subdirectory next to each input file.
  - **Implementation:** 
    - `cli.py` dispatches to `batch_runner.run_batch_processing`, `batch_runner.run_count_mode`, or `batch_runner.run_split_mode` based on the `--mode` argument.
    - `batch_runner.py` contains the new `run_count_mode` and `run_split_mode` functions with the respective logic.