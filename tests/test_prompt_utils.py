"""
Unit tests for the prompt_utils module.
"""

import os
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

from src.prompt_utils import load_system_prompt, load_prompts


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return {'examples_dir': 'examples/examples.txt'}


@pytest.fixture
def mock_examples_content():
    """Create mock examples file content for testing."""
    return """Example 1: This is a good example.
Example 2: This is another good example.
Example 3: This is a third example."""


@pytest.fixture
def mock_prompt_template():
    """Create a mock prompt template with placeholder."""
    return """You are a batch grader. Grade responses based on these examples:

{dynamic_examples}

Provide numerical scores only."""


@pytest.fixture
def mock_generic_prompt_template():
    """Create a mock generic prompt template."""
    return """You are a generic evaluator. Grade responses on a scale of 1-5.
Provide numerical scores only."""


def test_load_system_prompt_with_custom_examples(mock_config,
                                                 mock_examples_content,
                                                 mock_prompt_template):
    """Test loading a system prompt with custom examples."""
    with patch('src.prompt_utils.CONFIG_DIR', Path('/project')), \
         patch('src.prompt_utils.is_examples_file_default', return_value=False), \
         patch('pathlib.Path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_examples_content)), \
         patch('src.prompt_utils.load_prompt_template', return_value=mock_prompt_template):

        prompt = load_system_prompt(mock_config)

        # Should contain the template text and formatted examples
        assert "You are a batch grader" in prompt
        assert "- Example 1: This is a good example." in prompt
        assert "- Example 2: This is another good example." in prompt
        assert "- Example 3: This is a third example." in prompt
        assert "{dynamic_examples}" not in prompt  # Placeholder should be replaced


def test_load_system_prompt_with_default_examples(
        mock_config, mock_generic_prompt_template):
    """Test loading a system prompt with default examples."""
    with patch('src.prompt_utils.CONFIG_DIR', Path('/project')), \
         patch('src.prompt_utils.is_examples_file_default', return_value=True), \
         patch('src.prompt_utils.load_prompt_template', return_value=mock_generic_prompt_template):

        prompt = load_system_prompt(mock_config)

        # Should use the generic template
        assert prompt == mock_generic_prompt_template
        assert "You are a generic evaluator" in prompt


def test_load_system_prompt_missing_examples_file(mock_config):
    """Test that an error is raised when examples file is missing."""
    with patch('src.prompt_utils.CONFIG_DIR', Path('/project')), \
         patch('src.prompt_utils.is_examples_file_default', return_value=False), \
         patch('pathlib.Path.exists', return_value=False):

        with pytest.raises(FileNotFoundError) as excinfo:
            load_system_prompt(mock_config)

        assert "Examples file not found" in str(excinfo.value)


def test_load_system_prompt_empty_examples_file(mock_config,
                                                mock_prompt_template):
    """Test that an error is raised when examples file is empty."""
    with patch('src.prompt_utils.CONFIG_DIR', Path('/project')), \
         patch('src.prompt_utils.is_examples_file_default', return_value=False), \
         patch('pathlib.Path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data="")), \
         patch('src.prompt_utils.load_prompt_template', return_value=mock_prompt_template):

        with pytest.raises(ValueError) as excinfo:
            load_system_prompt(mock_config)

        assert "No examples found" in str(excinfo.value)


def test_load_system_prompt_missing_placeholder(mock_config,
                                                mock_examples_content):
    """Test that an error is raised when prompt template is missing the placeholder."""
    template_without_placeholder = "A template with no placeholders"

    with patch('src.prompt_utils.CONFIG_DIR', Path('/project')), \
         patch('src.prompt_utils.is_examples_file_default', return_value=False), \
         patch('pathlib.Path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=mock_examples_content)), \
         patch('src.prompt_utils.load_prompt_template', return_value=template_without_placeholder):

        with pytest.raises(ValueError) as excinfo:
            load_system_prompt(mock_config)

        assert "placeholder missing in prompt template" in str(excinfo.value)


def test_load_system_prompt_missing_config_key():
    """Test that an error is raised when 'examples_dir' is missing from config."""
    config_without_examples = {'other_key': 'value'}

    with pytest.raises(ValueError) as excinfo:
        load_system_prompt(config_without_examples)

    assert "'examples_dir' not found in config" in str(excinfo.value)


def test_load_prompts_file_not_found(tmp_path):
    """Test that FileNotFoundError is raised when prompts file doesn't exist."""
    nonexistent_file = tmp_path / "nonexistent.yaml"
    
    # Mock the logger to prevent TypeError with level comparison
    mock_logger = MagicMock()
    
    with patch('src.prompt_utils.logger', mock_logger), \
         pytest.raises(FileNotFoundError) as excinfo:
        load_prompts(nonexistent_file)
    assert f"Prompts file not found: {nonexistent_file}" in str(excinfo.value)


def test_load_prompts_invalid_yaml(tmp_path):
    """Test that YAMLError is raised when prompts file has invalid YAML."""
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("invalid: yaml: content: [missing bracket")
    
    # Mock the logger to prevent TypeError with level comparison
    mock_logger = MagicMock()
    
    with patch('src.prompt_utils.logger', mock_logger), \
         pytest.raises(yaml.YAMLError):
        load_prompts(invalid_yaml)


def test_load_prompts_invalid_format(tmp_path):
    """Test that YAMLError is raised when prompts file doesn't contain a dictionary."""
    invalid_format = tmp_path / "invalid_format.yaml"
    invalid_format.write_text("- this\n- is\n- a\n- list")
    
    # Mock the logger to prevent TypeError with level comparison
    mock_logger = MagicMock()
    
    with patch('src.prompt_utils.logger', mock_logger), \
         pytest.raises(yaml.YAMLError) as excinfo:
        load_prompts(invalid_format)
    assert "Expected a dictionary" in str(excinfo.value)


def test_load_prompts_success(tmp_path):
    """Test successful loading of prompts from a valid YAML file."""
    valid_yaml = tmp_path / "valid.yaml"
    expected_prompts = {
        "prompt1": "This is prompt 1",
        "prompt2": "This is prompt 2"
    }
    valid_yaml.write_text(yaml.dump(expected_prompts))
    
    # Mock the logger to prevent TypeError with level comparison
    mock_logger = MagicMock()
    
    with patch('src.prompt_utils.logger', mock_logger):
        loaded_prompts = load_prompts(valid_yaml)
    assert loaded_prompts == expected_prompts


def test_load_prompts_unexpected_error(tmp_path):
    """Test handling of unexpected errors during prompt loading."""
    valid_yaml = tmp_path / "valid.yaml"
    valid_yaml.write_text(yaml.dump({"test": "prompt"}))
    
    # Mock the logger to prevent TypeError with level comparison
    mock_logger = MagicMock()

    with patch('src.prompt_utils.logger', mock_logger), \
         patch('builtins.open', side_effect=Exception("Unexpected error")):
        with pytest.raises(Exception) as excinfo:
            load_prompts(valid_yaml)
        assert "Unexpected error" in str(excinfo.value)
