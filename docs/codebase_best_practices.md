# BatchGrader Codebase Best Practices

## Last updated: 2025-05-20

This document outlines best practices for maintaining and extending the BatchGrader codebase. It incorporates lessons learned from recent refactoring efforts and serves as a guide for consistent, maintainable code.

## 1. Import Structure

**NEVER use relative imports (e.g., `from .module import ...`) in this project. All imports must be absolute and resolvable from the project root. Do NOT use `src.*` style imports either.**

## NEVER use os for file access, it will fail

### Standard Import Order

Maintain a consistent order of imports in all Python modules:

1. **Standard library imports** (e.g., `os`, `sys`, `datetime`)
2. **Third-party package imports** (e.g., `pandas`, `openai`, `rich`)
3. **Local module imports** (internal BatchGrader modules)

```python
# Standard library imports
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

# Third-party imports
import pandas as pd
import openai
from rich.console import Console

# Local imports
from config_loader import load_config
from logger import logger
from token_tracker import log_token_usage_event
```

### Typing Imports

- Collect related typing imports on a single line when reasonable
- Prefer importing specific types rather than the entire module
- For larger type collections, use multiple lines with appropriate alignment

```python
# Good: Specific imports, grouped logically
from typing import Any, Dict, List, Optional, Union
from typing import Callable, TypeVar

# Avoid: Importing entire modules when only specific types are needed
import typing  # Not recommended unless you need many types
```

## 2. Preventing Circular Dependencies

Circular dependencies make code hard to understand, test, and maintain. They can also cause runtime errors that are difficult to debug.

### Warning Signs

- Import errors that mysteriously appear/disappear depending on import order
- Need to use delayed imports (`import` inside functions)
- `ImportError: cannot import name X` when X should be available

### Prevention Strategies

1. **Module Hierarchy**: Establish a clear hierarchy of modules where lower-level modules never import from higher-level ones

2. **Interface Modules**: Create dedicated interface modules that other modules can safely import from

3. **Dependency Injection**: Pass dependencies as parameters rather than importing them directly

4. **Function-level Imports**: For unavoidable cases, use imports inside functions (but document why)

   ```python
   def process_data():
       # Import here to avoid circular import with module_b
       from module_b import helper_function
       return helper_function()
   ```

## 3. Error Handling for Imports

Gracefully handle potential import issues, especially for optional dependencies.

```python
try:
    import optional_package
    OPTIONAL_PACKAGE_AVAILABLE = True
except ImportError:
    OPTIONAL_PACKAGE_AVAILABLE = False

def function_using_optional_feature():
    if not OPTIONAL_PACKAGE_AVAILABLE:
        logger.warning("Optional feature unavailable: optional_package not installed")
        return fallback_behavior()

    # Use optional_package here
```

## 4. Module Organization

### Single Responsibility Principle

Each module should have a clear, well-defined purpose. Break large modules into smaller, focused ones:

- `file_processor.py`: Handle file operations
- `token_utils.py`: Manage token counting and validation
- `llm_client.py`: Interface with LLM APIs

### Clear Module Documentation

Every module should start with a docstring explaining:

1. The module's purpose
2. Key functions/classes it provides
3. Usage examples
4. Dependencies and relationships with other modules

```python
"""
Token Tracker: Tracks and aggregates OpenAI API token usage.

- Daily aggregate for API limit compliance (legacy, output/token_usage_log.json)
- Per-request append-only event log (output/token_usage_events.jsonl)
- Aggregation and cost computation utilities

Event Schema (token_usage_event):
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

Cost is calculated using model pricing from docs/pricing.csv (per 1M tokens, input/output).
"""
```

## 5. Version Management

### Version Updates

When updating the version number:

1. Update in **all** relevant files:

   - `pyproject.toml` (both in `[project]` and `[tool.poetry]` sections)
   - `README.md` with a summary of changes
   - `docs/scratchpad.md` with detailed documentation
   - `uv.lock` file

2. Follow semantic versioning:

   - MAJOR: Incompatible API changes
   - MINOR: Add functionality in a backward-compatible manner
   - PATCH: Backward-compatible bug fixes

3. Include a detailed changelog in both:
   - Brief summary in README.md
   - Comprehensive documentation in docs/scratchpad.md

## 6. Testing Best Practices

### Comprehensive Test Coverage

- Write tests for all new functionality
- Fix failing tests promptly rather than disabling them
- Use mocking for external dependencies (e.g., OpenAI API)

### Test Structure

- Organize tests to mirror the source structure
- Use descriptive test names that explain what is being tested
- Group tests logically by functionality

```python
def test_token_counter_handles_empty_dataframe():
    # Test that the token counter gracefully handles empty dataframes

def test_token_counter_includes_system_prompt_in_count():
    # Test that the system prompt is correctly included in token counts
```

### Test Fixtures

- Use fixtures for common setup code
- Keep fixtures focused and small
- Document what each fixture provides

## 7. Constants and Configuration

- Centralize constants in `constants.py`
- Use environment variables for sensitive values (API keys)
- Provide clear default values for all configuration options
- Use type hints for configuration values

## 8. Error Handling and Logging

- Use appropriate logging levels (debug, info, warning, error)
- Include context in error messages
- Propagate errors to the appropriate level for handling
- Consider the user experience when designing error messages

## 9. Code Style and Documentation

- Follow consistent code formatting (use a linter/formatter, yapf & ty are currently used)
- Document complex logic with comments
- Keep functions small and focused
- Use type hints consistently throughout the codebase

## 10. Dependency Management

- Pin dependencies with specific versions
- Regularly update dependencies for security patches
- Document dependency requirements clearly
- Consider using dependency groups for optional features

By following these practices, we can maintain a clean, maintainable, and robust codebase that is easy to extend and modify.
