import os
import pytest
from pathlib import Path
import yaml
import tempfile
import logging
import shutil
from src.constants import PROJECT_ROOT
from src.config_loader import load_config, DEFAULT_CONFIG, ensure_config_files, is_examples_file_default, DEFAULT_EXAMPLES_TEXT


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                     delete=False) as tmp:
        yaml.dump(
            {
                "openai_model_name": "test-model",
                "custom_key": "custom_value"
            }, tmp)
        tmp_path = tmp.name

    yield tmp_path

    # Clean up
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def empty_config_file():
    """Create an empty config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                     delete=False) as tmp:
        tmp_path = tmp.name

    yield tmp_path

    # Clean up
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def invalid_config_file():
    """Create an invalid YAML config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                     delete=False) as tmp:
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
    assert config["max_simultaneous_batches"] == DEFAULT_CONFIG[
        "max_simultaneous_batches"]
    assert config["input_dir"] == DEFAULT_CONFIG["input_dir"]


def test_load_config_with_env_var():
    """Test that environment variables override config file values."""
    # Save original env var if it exists
    original_api_key = os.environ.get("OPENAI_API_KEY")

    try:
        # Set test environment variable
        os.environ["OPENAI_API_KEY"] = "test-api-key-from-env"

        # Load config with default path
        config = load_config()

        # Check that the API key from env var is used
        assert config["openai_api_key"] == "test-api-key-from-env"

    finally:
        # Restore original environment
        if original_api_key:
            os.environ["OPENAI_API_KEY"] = original_api_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)


def test_load_config_with_empty_file(empty_config_file):
    """Test loading config from an empty file (should use all defaults)."""
    config = load_config(empty_config_file)

    # Should contain all DEFAULT_CONFIG values
    for key, value in DEFAULT_CONFIG.items():
        assert config[key] == value


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
    from config_loader import DEFAULT_EXAMPLES_TEXT

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
    assert is_examples_file_default(
        tmp_path /
        "nonexistent.txt") is True  # Should return True for missing files


def test_ensure_config_files_creates_default_examples(caplog):
    """
    Test that ensure_config_files creates a default examples.txt if it's missing.
    This covers lines 89-92 of config_loader.py.
    """
    logger = logging.getLogger("test_config_creation")
    caplog.set_level(logging.INFO)

    examples_file_path = (PROJECT_ROOT / DEFAULT_CONFIG['examples_dir']).resolve()
    examples_file_backup_path = examples_file_path.with_suffix(examples_file_path.suffix + ".test_bak")

    file_existed_before_test = examples_file_path.exists()

    try:
        if file_existed_before_test:
            shutil.move(str(examples_file_path), str(examples_file_backup_path))

        # Action: Call ensure_config_files
        ensure_config_files(logger)

        # Assertions
        assert examples_file_path.exists(), f"Default examples file '{examples_file_path}' was not created."
        with open(examples_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        assert content == DEFAULT_EXAMPLES_TEXT, "Content of created examples file does not match default text."
        
        # Check for the specific log message related to examples.txt creation
        assert f"Default examples file created at '{examples_file_path}'." in caplog.text, "Log message for examples file creation not found."

    finally:
        # Teardown: Remove the examples file that was potentially created by ensure_config_files
        if examples_file_path.exists():
            os.remove(examples_file_path)

        # Restore the backup if it was made
        if file_existed_before_test and examples_file_backup_path.exists():
            shutil.move(str(examples_file_backup_path), str(examples_file_path))
        elif file_existed_before_test and not examples_file_backup_path.exists():
            # This case might occur if the backup move failed or the backup was unexpectedly removed.
            # For robustness, if the original file existed but backup is gone, try to recreate a default one to leave system in a known state.
            # Or, simply log a warning that restoration failed.
            logger.warning(f"Could not restore {examples_file_path} from {examples_file_backup_path} as backup was not found.")
