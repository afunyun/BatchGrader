"""
Unit tests for the evaluator module.
"""

import os
import pytest
import sys
import yaml
from unittest.mock import patch, mock_open

from src.evaluator import load_prompt_template


@pytest.fixture
def mock_prompts_yaml():
    """Create mock prompts.yaml content."""
    return """
evaluation_prompt: |
  You are an evaluator. Rate the following responses:
  Scale:
  5 - Excellent
  4 - Good
  3 - Average
  2 - Poor
  1 - Very poor
  
another_prompt: |
  This is another prompt template.
"""


@pytest.fixture
def mock_config_loader():
    """Create mock DEFAULT_PROMPTS from config_loader."""
    return {
        'evaluation_prompt': 'Default evaluation prompt from config_loader.',
        'batch_prompt': 'Default batch prompt from config_loader.'
    }


def test_load_prompt_template_success(mock_prompts_yaml):
    """Test loading a prompt template that exists in prompts.yaml."""
    with patch('builtins.open', mock_open(read_data=mock_prompts_yaml)):
        with patch(
                'yaml.safe_load',
                return_value=
            {
                'evaluation_prompt':
                'You are an evaluator. Rate the following responses:\nScale:\n5 - Excellent\n4 - Good\n3 - Average\n2 - Poor\n1 - Very poor\n',
                'another_prompt': 'This is another prompt template.'
            }):
            prompt = load_prompt_template('evaluation_prompt')
            assert prompt == 'You are an evaluator. Rate the following responses:\nScale:\n5 - Excellent\n4 - Good\n3 - Average\n2 - Poor\n1 - Very poor\n'


def test_load_prompt_template_missing_from_yaml_fallback_to_default():
    """Test fallback to DEFAULT_PROMPTS when prompt is missing from prompts.yaml."""
    with patch('builtins.open', mock_open(read_data="{}")):
        with patch('yaml.safe_load', return_value={}):
            with patch('src.evaluator.DEFAULT_PROMPTS',
                       {'evaluation_prompt': 'Default fallback prompt.'}):
                with patch(
                        'sys.stderr'):  # Capture stderr to avoid test output
                    prompt = load_prompt_template('evaluation_prompt')
                    assert prompt == 'Default fallback prompt.'


def test_load_prompt_template_file_not_found_fallback():
    """Test fallback to DEFAULT_PROMPTS when prompts.yaml cannot be loaded."""
    with patch('builtins.open', side_effect=FileNotFoundError()):
        with patch('src.evaluator.DEFAULT_PROMPTS',
                   {'evaluation_prompt': 'Default fallback prompt.'}):
            with patch('sys.stderr'):  # Capture stderr to avoid test output
                prompt = load_prompt_template('evaluation_prompt')
                assert prompt == 'Default fallback prompt.'


def test_load_prompt_template_fail_complete():
    """Test complete failure when prompt is not in prompts.yaml or DEFAULT_PROMPTS."""
    with patch('builtins.open', mock_open(read_data="{}")):
        with patch('yaml.safe_load', return_value={}):
            with patch('src.evaluator.DEFAULT_PROMPTS', {}):
                with patch('sys.stderr'):
                    with pytest.raises(RuntimeError) as excinfo:
                        load_prompt_template('missing_prompt')
                    assert "Failed to load prompt" in str(excinfo.value)
                    assert "should REALLY never happen" in str(excinfo.value)


def test_load_prompt_template_yaml_error():
    """Test handling of YAML parsing errors."""
    with patch('builtins.open', mock_open(read_data="invalid: yaml: content")):
        with patch('yaml.safe_load',
                   side_effect=yaml.YAMLError("YAML parsing error")):
            with patch('src.evaluator.DEFAULT_PROMPTS',
                       {'evaluation_prompt': 'Default fallback prompt.'}):
                with patch('sys.stderr'):
                    prompt = load_prompt_template('evaluation_prompt')
                    assert prompt == 'Default fallback prompt.'
