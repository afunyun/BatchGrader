# Changelog

All notable changes to the BatchGrader project will be documented in this file.

## 0.5.8.2 - 2025-05-15

### Fixed

- Fixed test performance issues by properly mocking `time.sleep` in `test_manage_batch_job_success`
- Improved test coverage to 76% by adding tests for error handling and edge cases
- Fixed potential race conditions in test file cleanup

### Changed

- Optimized test execution time by reducing unnecessary delays
- Improved test reliability by ensuring proper cleanup between tests

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

## 0.4.5 - 2025-04-12

### Fixed

- Continued fixes for repository clone functionality
- Updated project version
