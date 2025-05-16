# Code Quality Action Items ##PARTIALLY COMPLETED##

## 1. Global Configuration Management

- [x] Refactor code to avoid loading configuration at the module level (e.g., `config = load_config()` in `llm_client.py`, `log_utils.py`).
- [x] Pass the config object explicitly to classes/functions needing it (apply dependency injection).
- [x] Ensure tests can inject mock or temporary configs without cross-test contamination.

**Status**: Completed. Configuration is now passed explicitly using dependency injection.

---

## 2. Constants & Directory Path Handling

- [x] Never modify global constants (e.g., `LOG_DIR`, `ARCHIVE_DIR`) at runtime.  
       **Action:** Pass directory paths as parameters or include them in the configuration object.
- [x] Ensure all constant values are defined in a single location (e.g., `constants.py`) and remain immutable throughout runtime.

**Status**: Completed. Global constants are no longer modified at runtime. Instead, values are passed as parameters.

---

## 3. API Key Usage in `LLMClient`

- [x] Remove or clearly document the global assignment `openai.api_key = self.api_key` in `LLMClient`.
- [x] Use only the client instance (`OpenAI(api_key=...)`) for all API operations.
- [x] Prepare for possible use of multiple concurrent clients with different API keys.

**Status**: Completed. Removed global assignment and each thread now uses its own client instance.

---

## 4. Dependency Direction & Defaults

- [x] Consolidate default prompt definitions; avoid defining `DEFAULT_PROMPTS` in both `config_loader.py` and `evaluator.py`.
- [x] Centralize default data structures or pass them explicitly as needed.
- [x] Ensure module dependencies are unidirectional and clear.

**Status**: Completed. Default prompts are now imported from config_loader in evaluator.py.

---

## 5. Error Handling Consistency

- [x] In `input_splitter.py`, raise specific exceptions (not just return empty lists) for unrecoverable errors (e.g., missing `file_prefix`).
- [x] Let callers handle exceptions for greater robustness and clarity.

**Status**: Completed. Added proper exception classes and raised specific exceptions.

---

## 6. Test Behavior Isolation

- [x] Remove or refactor test-specific logic from production code (e.g., checks like `self.api_key == "TEST_KEY_FAIL_CONTINUE"` in `llm_client.py`).
- [x] Use mocking/patching in test suites to simulate failures or special behaviors.

**Status**: Completed. Removed test-specific logic from production code.

---

## 7. General Code Quality

- [ ] Ensure all functions/classes have comprehensive docstrings.
- [ ] Use `pathlib.Path` consistently for all file operations.
- [ ] Maintain and expand type hint coverage.
- [ ] Keep logging user-friendly and consistent (preferably using `BatchGraderLogger` with `RichHandler`).

**Status**: In progress. Some improvements made, but more comprehensive documentation and type hints needed.

---

## 8. Individual File/Line Action Items

- **src/batch_runner.py:171**  
  **[x]** _Bug risk:_ Do not modify global constants (`LOG_DIR`, `ARCHIVE_DIR`) at runtime.  
  **Action:** Pass values as parameters or via config.

- **src/batch_runner.py:216**  
  **[x]** _Bug risk:_ `process_file` in the loop is not exception-safe.  
  **Action:** Wrap `process_file` in try/except to avoid halting on a single file's failure.

- **src/batch_runner.py:47**  
  **[ ]** _Code refinement:_ Encoder acquisition is duplicated.  
  **Action:** Refactor encoder acquisition into a shared utility (e.g., `utils.get_encoder()`), use everywhere needed.

- **src/input_splitter.py:58**  
  **[x]** _Bug risk:_ Logger fallback uses `globals()['logger']`.  
  **Action:** Use `logging.getLogger(__name__)` for module logger fallback.

- **src/input_splitter.py:199**  
  **[ ]** _Performance:_ Avoid using `DataFrame.iterrows()` for chunking large DataFrames.  
  **Action:** Switch to `itertuples()` or vectorized operations for efficiency.

- **src/input_splitter.py:233**  
  **[x]** _Edge case:_ Both `token_limit` and `row_limit` can be `None`, causing unintended chunking.  
  **Action:** Raise `ValueError` if neither limit is set.

- **src/file_processor.py:551**  
  **[ ]** _Bug risk:_ Renaming 'id' to 'custom_id' may cause confusion if both exist.  
  **Action:** Handle the case where both columns are present; issue a warning or choose a clear schema.

- **src/file_processor.py:557**  
  **[ ]** _Bug risk:_ Blindly casting 'custom_id' to string may mask missing/duplicate IDs.  
  **Action:** Validate presence and uniqueness of 'custom_id' before casting.

- **src/file_processor.py:679**  
  **[x]** _Bug risk:_ Using a shared `LLMClient` with `ThreadPoolExecutor` may not be thread-safe.  
  **Action:** Ensure thread safety or provide each thread with its own client instance.

- **src/llm_utils.py:97**  
  **[ ]** _Bug risk:_ `with_data()` does not update `end_time` or `duration`.  
  **Action:** Update timing fields in `with_data` for consistency.

---

## Progress Summary

- **Completed**: 11 items
- **In Progress**: 7 items
- **Not Started**: 0 items

Major architectural improvements:

1. Removed module-level configuration loading
2. Implemented dependency injection
3. Fixed threading safety issues
4. Improved error handling
5. Removed hard-coded constants modification
6. Isolated test-specific code from production
