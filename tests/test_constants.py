"""
Unit tests for the constants module.
"""

from pathlib import Path

import pytest

from batchgrader.constants import (
    ARCHIVE_DIR,
    BATCH_API_ENDPOINT,
    DEFAULT_ARCHIVE_DIR,
    DEFAULT_BATCH_DESCRIPTION,
    DEFAULT_EVENT_LOG_PATH,
    DEFAULT_GLOBAL_TOKEN_LIMIT,
    DEFAULT_LOG_DIR,
    DEFAULT_MODEL,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_PRICING_CSV_PATH,
    DEFAULT_PROMPTS_FILE,
    DEFAULT_RESPONSE_FIELD,
    DEFAULT_SPLIT_TOKEN_LIMIT,
    DEFAULT_TOKEN_USAGE_LOG_PATH,
    LOG_DIR,
    MAX_BATCH_SIZE,
    PROJECT_ROOT,
)


@pytest.mark.parametrize(
    "subdir, expected_exists",
    [
        ("src", True),
        ("tests", True),
        ("config", True),
        ("nonexistent", False),
    ],
    ids=["src-exists", "tests-exists", "config-exists", "nonexistent-missing"],
)
def test_project_root_structure(subdir, expected_exists):
    # Arrange

    # Act
    expected = PROJECT_ROOT / subdir

    # Assert
    if expected_exists:
        assert expected.exists() and expected.is_dir()
    else:
        assert not expected.exists()


def test_log_directories():
    # Arrange

    # Act

    # Assert
    assert DEFAULT_LOG_DIR == PROJECT_ROOT / "output" / "logs"
    assert DEFAULT_ARCHIVE_DIR == DEFAULT_LOG_DIR / "archive"
    assert LOG_DIR == DEFAULT_LOG_DIR
    assert ARCHIVE_DIR == DEFAULT_ARCHIVE_DIR


@pytest.mark.parametrize(
    "value, expected_type, expected_positive",
    [
        (MAX_BATCH_SIZE, int, True),
        (DEFAULT_GLOBAL_TOKEN_LIMIT, int, True),
        (DEFAULT_SPLIT_TOKEN_LIMIT, int, True),
        (DEFAULT_POLL_INTERVAL, int, True),
    ],
    ids=[
        "max-batch-size", "global-token-limit", "split-token-limit",
        "poll-interval"
    ],
)
def test_positive_integer_constants(value, expected_type, expected_positive):
    # Arrange

    # Act

    # Assert
    assert isinstance(value, expected_type)
    if expected_positive:
        assert value > 0


def test_token_limits_relationship():
    # Arrange

    # Act

    # Assert
    assert DEFAULT_GLOBAL_TOKEN_LIMIT > DEFAULT_SPLIT_TOKEN_LIMIT


@pytest.mark.parametrize(
    "model, expect_nonempty, expect_dash, expect_digit",
    [
        (DEFAULT_MODEL, True, True, True),
        ("gpt-4o-2024-08-06", True, True, True),
        ("model-without-digits", True, True, False),
        ("", False, False, False),
    ],
    ids=["default-model", "realistic-model", "no-digit-model", "empty-model"],
)
def test_default_model_properties(model, expect_nonempty, expect_dash,
                                  expect_digit):
    # Arrange

    # Act

    # Assert
    assert isinstance(model, str)
    if expect_nonempty:
        assert model
    if expect_dash:
        assert "-" in model
    if expect_digit:
        assert any(char.isdigit() for char in model)
    else:
        assert not any(char.isdigit() for char in model)


@pytest.mark.parametrize(
    "response_field, expect_nonempty",
    [
        (DEFAULT_RESPONSE_FIELD, True),
        ("response", True),
        ("", False),
    ],
    ids=[
        "default-response-field", "custom-response-field",
        "empty-response-field"
    ],
)
def test_default_response_field(response_field, expect_nonempty):
    # Arrange

    # Act

    # Assert
    assert isinstance(response_field, str)
    if expect_nonempty:
        assert response_field
    else:
        assert not response_field


@pytest.mark.parametrize(
    "path_constant, expect_file, expect_parent",
    [
        (DEFAULT_TOKEN_USAGE_LOG_PATH, False, True),
        (DEFAULT_EVENT_LOG_PATH, False, True),
        (DEFAULT_PRICING_CSV_PATH, False, True),
        (DEFAULT_PROMPTS_FILE, False, True),
    ],
    ids=[
        "token-usage-log-path", "event-log-path", "pricing-csv-path",
        "prompts-file"
    ],
)
def test_path_constants_exist(path_constant, expect_file, expect_parent):
    # Arrange

    # Act

    # Assert
    assert isinstance(path_constant, Path)
    if expect_file:
        assert path_constant.exists()
    if expect_parent:
        assert path_constant.parent.exists()


@pytest.mark.parametrize(
    "endpoint, expect_nonempty",
    [
        (BATCH_API_ENDPOINT, True),
        ("/v1/chat/completions", True),
        ("", False),
    ],
    ids=["default-endpoint", "realistic-endpoint", "empty-endpoint"],
)
def test_batch_api_endpoint(endpoint, expect_nonempty):
    # Arrange

    # Act

    # Assert
    assert isinstance(endpoint, str)
    if expect_nonempty:
        assert endpoint
    else:
        assert not endpoint


@pytest.mark.parametrize(
    "desc, expect_nonempty",
    [
        (DEFAULT_BATCH_DESCRIPTION, True),
        ("Batch job for grading", True),
        ("", False),
    ],
    ids=[
        "default-batch-description", "custom-description", "empty-description"
    ],
)
def test_default_batch_description(desc, expect_nonempty):
    # Arrange

    # Act

    # Assert
    assert isinstance(desc, str)
    if expect_nonempty:
        assert desc
    else:
        assert not desc


@pytest.mark.parametrize(
    "prompts_file, valid_suffixes",
    [
        (DEFAULT_PROMPTS_FILE, (".txt", ".yaml", ".yml")),
        (Path("config/prompts.yaml"), (".yaml", ".yml")),
        (Path("config/prompts.txt"), (".txt", )),
    ],
    ids=["default-prompts-file", "yaml-prompts-file", "txt-prompts-file"],
)
def test_default_prompts_file(prompts_file, valid_suffixes):
    # Arrange

    # Act

    # Assert
    assert isinstance(prompts_file, Path)
    assert prompts_file.suffix in valid_suffixes


@pytest.mark.parametrize(
    "interval, expect_positive",
    [
        (DEFAULT_POLL_INTERVAL, True),
        (60, True),
        (0, False),
        (-1, False),
    ],
    ids=["default-poll-interval", "sixty", "zero", "negative"],
)
def test_poll_interval(interval, expect_positive):
    # Arrange

    # Act

    # Assert
    assert isinstance(interval, int)
    if expect_positive:
        assert interval > 0
    else:
        assert interval <= 0
