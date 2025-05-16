"""
Unit tests for the evaluator module.
"""

import os
import pytest
import sys
import yaml
from unittest.mock import patch, mock_open

from src.evaluator import load_prompt_template

# The actual default prompt from config_loader
DEFAULT_PROMPT_TEXT = (
    'You are an evaluator trying to determins the closeness of a response to a given style, examples of which will follow. Given the following examples, evaluate whether or not the response matches the target style.\n\n'
    'Examples:\n{dynamic_examples}\n\n'
    'Scoring should be as follows:\n'
    '5 - Perfect match\n'
    '4 - Very close\n'
    '3 - Somewhat close\n'
    '2 - Not close\n'
    '1 - No match\n\n'
    'Output only the numerical scores, one per line, in the same order as inputs.'
)


@pytest.fixture
def mock_prompts_yaml():
    """Create mock prompts.yaml content."""
    return """
batch_evaluation_prompt: |
  Please evaluate the following text and provide a score from 1 to 10 based on the provided examples.
  {dynamic_examples}
  Text: {input}
  Score:

batch_evaluation_prompt_generic: |
  You are an evaluator. Given the following message, rate its overall quality on a scale of 1 to 5.
  The scale is as follows:
  5 - Excellent
  4 - Good
  3 - Average
  2 - Poor
  1 - Very poor
  Output only the numerical score.
"""


def test_load_prompt_template_success(mock_prompts_yaml):
    """Test loading a prompt template that exists in prompts.yaml."""
    with patch('builtins.open', mock_open(read_data=mock_prompts_yaml)):
        with patch(
                'yaml.safe_load',
                return_value=
            {
                'batch_evaluation_prompt':
                'Please evaluate the following text and provide a score from 1 to 10 based on the provided examples.\n{dynamic_examples}\nText: {input}\nScore:',
                'batch_evaluation_prompt_generic':
                'You are an evaluator. Given the following message, rate its overall quality on a scale of 1 to 5.\nThe scale is as follows:\n5 - Excellent\n4 - Good\n3 - Average\n2 - Poor\n1 - Very poor\nOutput only the numerical score.'
            }):
            prompt = load_prompt_template('batch_evaluation_prompt')
            assert 'Please evaluate the following text' in prompt
            assert '{dynamic_examples}' in prompt
            assert '{input}' in prompt


def test_load_prompt_template_missing_from_yaml_fallback_to_default():
    """Test fallback to DEFAULT_PROMPTS when prompt is missing from prompts.yaml."""
    with patch('builtins.open', mock_open(read_data="{}")):
        with patch('yaml.safe_load', return_value={}):
            with patch('config_loader.DEFAULT_PROMPTS',
                       {'batch_evaluation_prompt': DEFAULT_PROMPT_TEXT}):
                with patch(
                        'sys.stderr'):  # Capture stderr to avoid test output
                    prompt = load_prompt_template('batch_evaluation_prompt')
                    assert prompt == DEFAULT_PROMPT_TEXT


def test_load_prompt_template_file_not_found_fallback():
    """Test fallback to DEFAULT_PROMPTS when prompts.yaml cannot be loaded."""
    with patch('builtins.open', side_effect=FileNotFoundError()):
        with patch('config_loader.DEFAULT_PROMPTS',
                   {'batch_evaluation_prompt': DEFAULT_PROMPT_TEXT}):
            with patch('sys.stderr'):  # Capture stderr to avoid test output
                prompt = load_prompt_template('batch_evaluation_prompt')
                assert prompt == DEFAULT_PROMPT_TEXT


def test_load_prompt_template_fail_complete():
    """Test complete failure when prompt is not in prompts.yaml or DEFAULT_PROMPTS."""
    with patch('builtins.open', mock_open(read_data="{}")):
        with patch('yaml.safe_load', return_value={}):
            with patch('config_loader.DEFAULT_PROMPTS', {}):
                with patch('sys.stderr'):
                    with pytest.raises(RuntimeError) as excinfo:
                        load_prompt_template('missing_prompt')
                    assert "Failed to load prompt" in str(excinfo.value)
                    assert "should REALLY never happen" in str(excinfo.value)


def test_load_prompt_template_yaml_error():
    """Test handling of YAML parsing errors."""
    with patch('pathlib.Path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data="invalid: yaml: content")), \
         patch('yaml.safe_load', side_effect=yaml.YAMLError("Invalid YAML")), \
         patch('config_loader.DEFAULT_PROMPTS', {'batch_evaluation_prompt': DEFAULT_PROMPT_TEXT}), \
         patch('sys.stderr'):
        prompt = load_prompt_template('batch_evaluation_prompt')
        assert prompt == DEFAULT_PROMPT_TEXT


def test_load_prompt_template_general_error():
    """Test handling of general exceptions during prompt loading."""
    with patch('pathlib.Path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data="valid: yaml")), \
         patch('yaml.safe_load', side_effect=Exception("Unexpected error")), \
         patch('config_loader.DEFAULT_PROMPTS', {'batch_evaluation_prompt': DEFAULT_PROMPT_TEXT}), \
         patch('sys.stderr'):
        prompt = load_prompt_template('batch_evaluation_prompt')
        assert prompt == DEFAULT_PROMPT_TEXT
