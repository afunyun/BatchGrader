**0.4.3 (2025-05-12):**

- Fixed pytest.ini_options not being read by pytest.

**0.4.2 (2025-05-12):**

- Fixed asyncio_default_fixture_loop_scope not being read by pytest.

**0.4.1 (2025-05-11):**

- Fixed tests returning None for vars and not knowing what dir they were in, that was rough.

**0.4.0 (2025-05-11):**

- Tightened token limits, reintroduced exception when examples file is missing, revamped test runner, unified file paths, updated & restructured docs.
- No evidence of logger state being manually manipulated so no issues there.

**0.3.3 (2025-05-11):**

- Import Cleanup & CLI Table Fixes: nuked unused imports, pointed all test outputs properly at tests/output/, and swapped prints to console for RichJobTable live-updating wobbly progress bar. Forgot to give a version number so it gets a fake one 0.3.3.

**0.3.2 (2025-05-11):**

- Now blows up if pricing.csv is gone. System will now do a recursive deep-merge of configs instead of a shallow one so you don't lose nested settings.

**0.3.1 (2025-05-11):**

- Chunking finally works and storage paths have been cleaned.

**0.3.0 (2025-05-11):**

- Ditched raw prints for rich console output with colors & emoji, added a summary table for jobs/tokens/cost, and beefed up logging.

# [0.2.12] (2025-05-11)

- Fully integrated the [rich](https://rich.readthedocs.io/) library for a live-updating, colorized CLI job status table.
- Batch job polling and progress/status updates now appear in a single, clean table (no more repeated print statements).
- All job statuses, errors, and progress are visible at a glance; failures are highlighted.
- Logging remains robust: persistent logs are still written to file for debugging and traceability.
- Yeeted straggling print statements in llm_client.py
- See `docs/scratchpad.md` for rationale and implementation notes.

## [0.2.11] (2025-05-11)

- Added chunking subdir to prune extra files after processing

- only pruned once processing is complete - the full files contain the content so there is no information loss by doing so.
- Logger now always writes a clear startup message to every log file as soon as it is instantiated, guaranteeing that all log files contain at least one entry.
- LOG PRUNING/ARCHIVING
- To prevent unbounded growth of the output/logs/ directory, prune/rotate logs before logger instantiation
- Moves oldest logs to output/logs/archive/ if threshold exceeded, and deletes oldest archive logs if needed
- See docs/scratchpad.md for rationale and changelog
- This improves traceability and debugging, especially for CI/test runs or troubleshooting failed jobs.
- Logger always writes a startup message to avoid empty logs
- Logging colour coding + console output is working perfectly
- All error/exit paths now log to file before exiting w/ new logger
- Tests seem to be functioning, CLI args working properly, tests succeeding
- Directory structure has been repaired & is standard now.

## [0.2.10] (2025-05-11)

- Renamed `tests/test_inputs/` → `tests/input/` and `tests/test_outputs/` → `tests/output/`.
- Added `.keep` files to ensure important directories exist in both production and test environments.
- Updated all test configs and code references to match the new convention.
- Added `.gitignore` to reflect the new test output directory.
- Tests 3/5 are complete and functional. Rerun is in order due to changes and hopefully all 5 will run.

## [0.2.9] (2025-05-11)

- Added --config arg, stress tested --file cli arg.
- Created testing agenda and a testing script + multiple faulty configs/datasets for testing purposes.

## [0.2.8] (2025-05-11)

- Integration: Concurrent Processing in Main Workflow
  - Integrated concurrent batch processing into the main workflow (`process_file`).
  - Now, if forced chunking or token-based splitting is needed, `process_file` routes to `process_file_concurrently` for parallel execution.
  - Otherwise, it uses the legacy single-batch logic for backward compatibility.
- All results and errors are saved and logged as before.
- The system is now ready for targeted testing of both single-batch and concurrent batch modes.
- Implementation of the concurrent batch job system has begun.
- Config layer (config_loader.py, config.yaml) and chunk job abstraction (batch_job.py) are complete.
- The `_generate_chunk_job_objects` helper is implemented and tested for both forced and token-based chunking.

### [0.2.7] (2025-05-11)

> **Note:** Implementation of concurrent batch (multi-job) processing is underway! Soon, BatchGrader will be able to split large datasets into multiple jobs and process them in parallel for much greater speed and scalability. Version 0.3.0 should contain this, assuming everything doesn't explode.

### [0.2.6] (2025-05-11)

- Large update to `token_tracker.py` to allow for tracking token usage and cost over time

- Why? if no info i am sad. Also the prices are low enough that if you're not running massive batches you're essentially always seeing 0 for the cost - want to be able to track cost even over a large number of smaller runs. Also added additional methods to `batch_runner.py` to display cumulative statistics. Added CLI arguments for this functionality (toggle/override). Statistics will not appear on unsuccessful runs OR splitting/counting runs. Only successful executions with a response receive get added to the new log and trigger the new stats display. Updated error handling specifically for CLI args - if positioning is incorrect, has more useful error message.

### [0.2.5] (2025-05-11)

- Added `batch_runner.py` to allow for use of the command I provided XD

- Update pyproject version because it was a crisp 5 "versions" (updates) behind. Prep for update check method because I want to be able to check for updates. I change things at random and frequently like a boss.

### [0.2.4] (2025-05-11)

- Fixed tiktoken error output to actually raise an exception

- this is instead of printing an error to console and then exiting. Deleted redundant docs images (NOOOOO). Update console output to provide more information about the current runtime while the `batch_runner.py` is running - previously it was fairly hard to see what options were actually running. Added `split_token_limit` to config.yaml to allow for splitting input files into chunks that do not exceed the configured token limit. CLI and normal operation now respect this limit rather than the Token Limit if it is applicable.

### [0.2.3] (2025-05-11)

- Release fixes - everything seems to ...work... on my machine at least

- Testing leaves something to be desired. Also managed to annihilate the branches for a second.

### [0.2.2] (2025-05-11)

- Fix multiple input files continuing to run/submit jobs if a reply comes back with errors. (preliminary fix not fully tested)

### [0.2.1] (2025-05-11)

- fixed cli args which i managed to put in horribly broken, nice, unified loading logic as well for args
