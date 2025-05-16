# Changelog

All notable changes to the BatchGrader project will be documented in this file.

## 0.5.9 - 2025-05-16

### Added

- Completed test coverage improvements across multiple modules
- Achieved 100% coverage for logger.py
- Near-perfect coverage (>90%) for cost_estimator.py, config_loader.py, evaluator.py, log_utils.py, token_tracker.py, and utils.py
- All tests now running with pytest, avoiding actual API calls to OpenAI
- Updated code_review.md to reflect completed test improvements

### Changed

- Standardized testing patterns across the codebase
- Improved test reliability with proper resource cleanup
- Enhanced test assertions for better debugging

## 0.5.8.3 - 2025-05-16

### Added

- Added comprehensive tests for the `input_splitter` module ([`src/input_splitter.py`](src/input_splitter.py:1)), covering various file types, splitting strategies, edge cases, and error handling. Test file created at [`tests/test_input_splitter.py`](tests/test_input_splitter.py).

## 0.5.8.2 - 2025-05-15

### Fixed

- Fixed test performance issues by properly mocking `time.sleep` in `test_manage_batch_job_success`
- Improved test coverage to 76% by adding tests for error handling and edge cases
- Fixed potential race conditions in test file cleanup
- Fixed date-related tests in token_tracker.py
- Enhanced mock setup for file operations
- Improved consistency in encoder mocks

### Changed

- Optimized test execution time by reducing unnecessary delays
- Improved test reliability by ensuring proper cleanup between tests
- Standardized testing patterns across the codebase
- Added proper resource cleanup in tests to prevent side effects
- Enhanced test assertions for better debugging and reliability

## 0.5.8.1 - 2025-05-16

### Fixed

- Fixed project root path detection to work correctly in different environments
- Updated config file path resolution to use the correct project root
- Fixed test failures caused by incorrect path resolution
- Made the test infrastructure more robust by adding src to the Python path
- Added default config example files to ensure proper application initialization

### Added

- Created example config files for easier setup
- Improved project root detection logic to handle various deployment scenarios
- Added more robust path handling across different operating systems

## 0.5.8 - 2025-05-16

### Fixed

- Fixed all remaining test failures (139/139 tests now passing with 70% coverage)
- Fixed import paths in test files to avoid 'src.' prefix
- Removed/commented defunct make_request tests in test_llm_client.py
- Updated BatchJob tests to match actual implementation
- Fixed test_process_dataframe_with_llm in test_file_processor.py

### Added

- Improved documentation and code reliability across multiple modules

## 0.5.7.3 - 2025-05-16

### Fixed

- Fixed last 8 failing tests and removed reliance on path mocking in tests
- Bumped package version in pyproject.toml and uv.lock

### Added

- Updated documentation in README.md and scratchpad.md to reflect the new version

## 0.5.7.1 - 2025-05-16

### Fixed

- Fixed date-related tests in token_tracker.py by updating mock datetime values and import paths
- Enhanced mock setup for file operations in test_file_utils.py
- Improved consistency in encoder mocks throughout test_file_processor.py

### Added

- Standardized testing patterns across the entire codebase
- Added proper resource cleanup in tests to prevent side effects
- Fixed and enhanced test assertions for better debugging and reliability

## 0.5.7 - 2025-05-16

### Changed

- Replaced all relative imports in src/ modules with absolute imports
- Ensured all test files use direct (top-level) imports
- Updated codebase to enforce: never use relative imports or src.\* imports; all imports must be absolute and resolvable from the project root
- Updated documentation and best practices accordingly

## 0.5.6 - 2025-05-16

### Added

- Comprehensive code review and refactoring addressing items from docs/code_review.md
- Added progress tracking to BatchJob
- Centralized configuration for token_tracker.py (moved paths to constants.py)
- Standardized logging across the application using a new setup_logging function and logging.getLogger(**name**) in modules
- Improved docstrings and added comments in key modules (file_processor.py, token_tracker.py, logger.py)

### Fixed

- Reviewed and confirmed secure API key handling
- Made try-except blocks more specific in batch_runner.py and cli.py
- Migrated os.path usage to pathlib.Path in several modules
- Improved type hint coverage in token_tracker.py, input_splitter.py, and utils.py
- Resolved various linter errors
- Reduced code duplication through prior refactoring

## 0.5.5 - 2025-05-16

### Fixed

- Fixed evaluator error message to match test expectations
- Updated LLMClient mock to properly handle config parameter
- Fixed failing tests for better stability

## 0.5.4 - 2025-05-16

### Added

- Implemented comprehensive dependency injection across the codebase
- Enhanced test isolation by removing test-specific logic from production code
- Standardized configuration management with proper parameter passing

### Fixed

- Fixed thread safety issues in concurrent file processing
- Removed global state and module-level configuration loading
- Improved exception handling with specific exception types

## 0.5.3 - 2025-05-16

### Added

- Standardized import structures throughout the codebase
- Added more tests and ensured they are passing

### Fixed

- Resolved circular dependency issues
- Enhanced error handling for import operations
- Improved robustness across all modules

## 0.5.1 - 2025-05-16

### Added

- Enhanced test suite reliability with comprehensive tiktoken mocking
- Added proper logger implementation in input_splitter.py

### Fixed

- Fixed DataFrame comparison issues in tests for more robust equality checks
- Improved error handling in test framework
- Standardized test fixtures and mock implementations

## 0.5.0 - 2025-05-12

### Added

- Major architectural refactoring for better modularity
- Split core functionality into focused modules
- Added operational modes: batch, count, and split
- Centralized configuration and constants
- Improved error handling and logging
- Enhanced test coverage and organization

## 0.4.5 - 2025-05-12

### Fixed

- Fixed repository clone functionality
- Updated project version to 0.4.5
- Fixed environment setup and import issues
- Removed local-only batch_runner.py from project root
- Updated run instructions for clarity

## 0.4.4 - 2025-05-12

### Fixed

- Fixed pytest.ini_options not being read by pytest
- Fixed asyncio_default_fixture_loop_scope not being read by pytest
- Fixed tests returning None for variables and directory detection

### Changed

- Updated rich_display imports
- Improved error handling for test environment setup

## 0.4.3 - 2025-05-12

### Fixed

- Fixed pytest configuration issues
- Resolved test environment setup problems

## 0.4.2 - 2025-05-12

### Fixed

- Fixed asyncio fixture scope issues in tests
- Improved test reliability

## 0.4.1 - 2025-05-11

### Fixed

- Fixed test variable handling and directory detection
- Resolved test environment setup issues

## 0.4.0 - 2025-05-11

### Added

- Tightened token limits for better resource management
- Revamped test runner
- Unified file paths across the application
- Restructured documentation

### Fixed

- Reintroduced exception when examples file is missing
- Ensured proper logger state management

## 0.3.3 - 2025-05-11

### Changed

- Performed comprehensive import cleanup
- Standardized test output directories to tests/output/
- Replaced console prints with RichJobTable for better visualization
- Added live-updating progress bar

## 0.3.2 - 2025-05-11

### Added

- Added check for pricing.csv existence
- Implemented recursive deep-merge for configurations

### Fixed

- Fixed configuration merging to preserve nested settings

## 0.3.1 - 2025-05-11

### Fixed

- Fixed chunking functionality
- Cleaned up storage paths
- Improved file organization

## 0.3.0 - 2025-05-11

### Added

- Integrated Rich library for enhanced console output
- Added colorized CLI job status table
- Implemented summary table for jobs/tokens/cost
- Enhanced logging capabilities

### Changed

- Replaced raw print statements with Rich-based output
- Improved console visualization with emoji and colors
- Beefed up logging throughout the application

## 0.2.12 - 2025-05-11

### Added

- Live-updating, colorized CLI job status table
- Centralized job status, errors, and progress tracking
- Persistent logging to file for debugging

### Fixed

- Removed remaining print statements in llm_client.py
- Improved job status visibility

## 0.2.11 - 2025-05-11

### Added

- Chunking subdirectory for temporary files
- Log pruning and archiving system
- Automatic log rotation with archiving
- Startup message in every log file
- Color-coded console output

### Fixed

- Directory structure standardization
- Error handling for CLI arguments
- Test reliability

## 0.2.10 - 2025-05-11

### Changed

- Renamed test directories:
  - `tests/test_inputs/` → `tests/input/`
  - `tests/test_outputs/` → `tests/output/`
- Added `.keep` files to maintain directory structure
- Updated test configurations and references
- Modified `.gitignore` for new output directories

## 0.2.9 - 2025-05-11

### Added

- `--config` command line argument
- Testing script with multiple test cases
- Faulty configs/datasets for testing
- Testing agenda

## 0.2.8 - 2025-05-11

### Added

- Concurrent batch processing in main workflow
- `process_file_concurrently` function
- Config layer for batch processing
- Chunk job abstraction (`batch_job.py`)
- `_generate_chunk_job_objects` helper

### Changed

- Main workflow now supports both single and concurrent modes
- Backward compatibility with legacy single-batch logic

## 0.2.7 - 2025-05-11

### Added

- Initial implementation of concurrent batch processing
- Documentation for upcoming parallel processing features

### Changed

- Prepared codebase for parallel job execution
- Improved error handling for concurrent operations

## 0.2.6 - 2025-05-11

### Added

- Token usage and cost tracking over time
- Cumulative statistics display
- CLI arguments for statistics toggling

### Fixed

- Improved error handling for CLI arguments
- Enhanced cost calculation accuracy

## 0.2.5 - 2025-05-11

### Added

- `batch_runner.py` for command-line execution
- Version update checking mechanism

### Changed

- Updated project versioning approach
- Improved command-line interface

## 0.2.4 - 2025-05-11

### Added

- `split_token_limit` configuration
- Enhanced runtime information display

### Fixed

- Tiktoken error handling
- Console output formatting
- Token limit enforcement

## 0.2.3 - 2025-05-11

### Fixed

- Initial release stabilization
- Basic functionality verification
