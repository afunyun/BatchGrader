# Consolidated Code Improvement & Task List

## CURRENT PRIORITY

| Priority | Task                                         | Rationale                      | Status  |
| -------- | -------------------------------------------- | ------------------------------ | ------- |
| 1        | DONE! Fix chunking logic (split_token_limit) | Data integrity, core logic     | ✅      |
| 2        | DONE! Restore examples file existence check  | User-facing reliability        | ✅      |
| 3        | DONE! Logger handler management              | Traceability, debugging        | ✅      |
| 4        | Automated test improvements (Section III)    | Prevent regression, robustness | ⏳      |
| 5        | Code Quality                                 | Prevent regression, robustness | ✅      |
| 6        | Documentation/nitpicks                       | Prevent regression, robustness | ⏳      |

## I. Project Structure & Maintainability

    Extract Utilities Module:
        Suggestion: Consider extracting the rich display and logging setup into a dedicated utilities module. This would help reduce code duplication and improve overall maintainability.
    Refactor CLI Argument Parsing:
        Context: The command-line interface argument parsing in batch_runner.py is getting complicated.
        Suggestion: Refactor this into a dedicated function or class to enhance readability and make future extensions easier.

## II. Core Logic & Bug Fixes

    ✅ Logger Handlers Management (bug_risk):
        Issue: Manual flushing and closing of logger handlers can interfere with logger reuse across different modules, potentially breaking them if they expect the logger to remain active.
        Recommendation: Avoid manual flushing/closing. Rely on the logger's own lifecycle management or remove such cleanup steps.
    ✅ *DONE* Restore Existence Check for Examples File (bug_risk):
        Issue: A removed check for the existence of abs_examples_path might lead to unclear errors if the file is missing.
        Recommendation: Restore the existence check for abs_examples_path and raise a clear FileNotFoundError if it's not found (unless this is definitively handled elsewhere).
    ✅ *DONE* Correct Chunking Logic (bug_risk):
        # Issue: The current logic for chunking based on split_token_limit might not align with user intent.
        # Recommendation: The condition appears reversed. Use total_tokens > split_token_limit to trigger chunking when the number of tokens exceeds the specified limit.
    ✅ *DONE* Configuration Merging (bug_risk):
        # Issue: Merging user configuration with DEFAULT_CONFIG using update() performs only a shallow merge. This can lead to incorrect overrides of nested dictionaries.
        # Recommendation: Switch to a deep merge mechanism to ensure nested configuration keys are merged correctly.
    ✅ *DONE* Pricing CSV Handling:
        # Issue: A removed check for the existence of abs_examples_path might lead to unclear errors if the file is missing.
        # Recommendation: Restore the existence check for abs_examples_path and raise a clear FileNotFoundError if it's not found (unless this is definitively handled elsewhere).

## III. Testing Enhancements

⏳  Automated Assertions Beyond Exit Codes (testing):
        Suggestion: Enhance the test runner by adding automated checks that go beyond simple exit codes.
        Details: Verify output file contents (e.g., row counts, specific error messages), check for expected log messages, or validate processed-row counts. This will make tests more robust and reduce the need for manual validation.
        *Progress (2025-05-11):* `pytest` framework adopted. Initial tests for `LLMClient` now implement mocking of OpenAI API calls and assert specific content/error messages from batch outputs, representing a significant step towards this goal.
    Test Case for halt_on_chunk_failure: False (testing):
        Suggestion: Add a specific test case where halt_on_chunk_failure is set to False.
        Scenario: Use an input file where one chunk is designed to fail while others should succeed. This test will ensure that processing correctly continues for non-failing chunks when this setting is disabled.
        Example Test Config:

        {
            'name': 'Continue on Chunk Failure',
            'input': 'tests/input/chunk_with_failure.csv',
            'config': 'tests/test_config_continue_on_failure.yaml',
            'halt_on_chunk_failure': False,
        }

    Integration Tests for Configuration Files (testing):
        Suggestion: Add integration test cases in run_all_tests.py that execute batch_runner.py with invalid, broken, or incomplete config.yaml files.
        Goal: Verify the application's startup robustness and ensure proper error handling for issues like missing keys, invalid paths, and incorrect data types in configurations.

## IV. Code Quality & Specific Function Refactoring

✅   Refactor process_file_concurrently (code-quality):
        Issue: Low code quality score (23%) due to method length, cognitive complexity, and working memory.
        Recommendations:
            Reduce function length by extracting parts into smaller, dedicated functions (ideally < 10 lines per function).
            Minimize nesting, possibly by using guard clauses for early returns.
            Ensure variables have tight scopes, keeping related concepts grouped.
            Apply specific suggestions:
                Use named expressions to simplify assignments and conditionals (use-named-expression).
                Extract duplicate code into a separate function (extract-duplicate-method).
✅   Refactor LLMClient._process_batch_outputs (code-quality):
        Issue: Very low code quality score (9%) due to similar reasons as above. (Note: Some content related to retrieved_batch, last_status, time.sleep(self.poll_interval) appeared incomplete in the source).
        Recommendations:
            Make the function shorter and more readable.
            Extract functionality into separate, smaller functions.
            Reduce nesting (e.g., with guard clauses).
            Ensure tight variable scoping.
            Apply specific suggestions:
                Use named expressions (use-named-expression).
                Replace if statements with if expressions where appropriate (assign-if-exp).
                Explicitly raise exceptions from a previous error (raise-from-previous-error).

## V. General Code Quality & Style Improvements (code-quality)

✅ Simplify sum() Calls:
        Suggestion (simplify-constant-sum): Leverage the fact that sum() treats True as 1 and False as 0.
        Affected code (examples):
            succeeded = sum(bool(getattr(j, 'status', None) == 'completed') ...)
            failed = sum(bool(getattr(j, 'status', None) == 'failed') ...)
            errored = sum(bool(getattr(j, 'status', None) == 'error') ...)
✅ Control Flow and Expressions:
        Suggestion (reintroduce-else): Lift code into an else block after a jump in control flow (e.g., return, break, continue).
        Suggestion (assign-if-exp): Replace if statements with if expressions for assignments where suitable.
        Suggestion (boolean-if-exp-identity): Simplify boolean if expressions (e.g., if condition: return True else: return False to return condition).
✅ Remove Unnecessary Casts (remove-unnecessary-cast):
        Suggestion: Remove redundant casts to int, str, float, or bool where the type is already correct.
        (Note: A point regarding if end_date and dt.date() > datetime.fromisoformat(end_date).date(): and its conditions appeared incomplete in the source but likely relates to these general improvements or specific function refactoring.)

## VI. Typos & Nitpicks

✅   Filename Typo (nitpick/typo):
        Issue: Typo in test input filename: "corrupt.csv".
        Fix: Rename to "corrupt.csv" and update its reference in test_cases.
            'input': 'tests/input/corrupt.csv'
✅   Capitalization (typo):
        Issue: 'only' should be 'Only' at the beginning of a bullet point sentence.
        Example: "- only pruned once processing is complete..." should be "- Only pruned..."

## VII. Documentation & README

## TODO: Update README.md once all items addressed
