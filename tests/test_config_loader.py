import logging
import os
import shutil
import tempfile

import pytest
import yaml

from batchgrader.config_loader import (
    DEFAULT_CONFIG,
    DEFAULT_EXAMPLES_TEXT,
    ensure_config_files,
    is_examples_file_default,
    load_config,
)
from batchgrader.constants import PROJECT_ROOT


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(
            {"openai_model_name": "test-model", "custom_key": "custom_value"}, tmp
        )
        tmp_path = tmp.name

    yield tmp_path

    # Clean up
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def empty_config_file():
    """Create an empty config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp_path = tmp.name

    yield tmp_path

    # Clean up
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def invalid_config_file():
    """Create an invalid YAML config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        tmp.write("this is not: valid: yaml: content:")
        tmp_path = tmp.name

    yield tmp_path

    # Clean up
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


def test_load_config_with_custom_file(temp_config_file):
    """Test loading config from a custom file path."""
    config = load_config(temp_config_file)

    # Check custom values are loaded
    assert config["openai_model_name"] == "test-model"
    assert config["custom_key"] == "custom_value"

    # Check default values are preserved
    assert (
        config["max_simultaneous_batches"] == DEFAULT_CONFIG["max_simultaneous_batches"]
    )
    assert config["input_dir"] == DEFAULT_CONFIG["input_dir"]


def test_load_config_with_env_var(monkeypatch):
    """Test that environment variables override config file values."""
    # Set test environment variable
    test_api_key = "test-api-key-from-env"
    monkeypatch.setenv("OPENAI_API_KEY", test_api_key)

    # Load config with default path
    config = load_config()

    # Check that the API key from env var is used
    assert config["openai_api_key"] == test_api_key


def test_load_config_with_empty_file(empty_config_file):
    """Test loading config from an empty file (should use all defaults)."""
    config = load_config(empty_config_file)

    # Verify all default values are present and correct
    assert config == {**DEFAULT_CONFIG, **config}


def test_load_config_file_not_found():
    """Test that ValueError is raised when config file doesn't exist."""
    with pytest.raises(ValueError):
        load_config("/path/to/nonexistent/config.yaml")


def test_load_config_invalid_yaml(invalid_config_file):
    """Test that RuntimeError is raised when config file contains invalid YAML."""
    with pytest.raises(RuntimeError):
        load_config(invalid_config_file)


def test_is_examples_file_default(tmp_path):
    """Test is_examples_file_default function."""
    from batchgrader.config_loader import DEFAULT_EXAMPLES_TEXT

    # Create a file with default content
    default_file = tmp_path / "default.txt"
    with open(default_file, "w") as f:
        f.write(DEFAULT_EXAMPLES_TEXT)

    # Create a file with custom content
    custom_file = tmp_path / "custom.txt"
    with open(custom_file, "w") as f:
        f.write("This is custom content")

    assert is_examples_file_default(default_file) is True
    assert is_examples_file_default(custom_file) is False
    assert (
        is_examples_file_default(tmp_path / "nonexistent.txt") is True
    )  # Should return True for missing files


def test_ensure_config_files_creates_default_examples(caplog, monkeypatch, tmp_path):
    """
    Test that ensure_config_files creates a default examples.txt if it's missing.
    This covers lines 89-92 of config_loader.py.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a temporary directory for this test
    test_config_dir = tmp_path / "config"
    test_config_dir.mkdir()

    # Patch the PROJECT_ROOT to use our test directory
    monkeypatch.setattr(
        "batchgrader.config_loader.PROJECT_ROOT", test_config_dir)

    # Import here after monkeypatching
    from batchgrader.config_loader import ensure_config_files, DEFAULT_CONFIG

    examples_file_path = (
        test_config_dir / DEFAULT_CONFIG["examples_dir"]).resolve()

    # Ensure file doesn't exist before test
    examples_file_path.unlink(missing_ok=True)

    # Action: Call ensure_config_files
    ensure_config_files(logger)

    # Assertions
    assert (
        examples_file_path.exists()
    ), f"Default examples file '{examples_file_path}' was not created."
    with open(examples_file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    assert (
        content == DEFAULT_EXAMPLES_TEXT
    ), "Content of created examples file does not match default text."

    # Check for the specific log message related to examples.txt creation
    assert (
        f"Default examples file created at '{examples_file_path}'." in caplog.text
    ), "Log message for examples file creation not found."

    # No cleanup needed - tmp_path is automatically cleaned up by pytest
