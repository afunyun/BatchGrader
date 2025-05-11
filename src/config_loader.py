import yaml
from pathlib import Path
import os

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'
CONFIG_PATH = CONFIG_DIR / 'config.yaml'
PROMPTS_PATH = CONFIG_DIR / 'prompts.yaml'

DEFAULT_CONFIG = {
    'max_simultaneous_batches': 2,  # TESTING Number of parallel batch jobs per input file (concurrent chunk processing)
    'force_chunk_count': 0,         # TESTING If >1, forcibly split input into this many chunks regardless of token limits (for speed)
    'halt_on_chunk_failure': True,  # TESTING If True, aborts remaining chunks for a file if any chunk fails critically
    'input_dir': '../input',
    'output_dir': '../output',
    'examples_dir': '../examples/examples.txt',
    'openai_model_name': 'gpt-4o-mini-2024-07-18',
    # system will pull from environment variables FIRST if it's set there.
    # openai_api_key: YOUR_OPENAI_API_KEY_HERE
    'poll_interval_seconds': 60,
    'max_tokens_per_response': 1000,
    'response_field': 'response',
    'batch_api_endpoint': '/v1/chat/completions',
    'token_limit': 2_000_000,
    'split_token_limit': 500_000,  # Max tokens per split file (default ~500k)
    'split_row_limit': None        # Max rows per split file (optional, default: unlimited)
}

"""
Event Dictionary (Table 1)
-------------------------
| Event Name                | Payload Schema                        | Description                         |
|-------------------------- |---------------------------------------|-------------------------------------|
| input_split_config_loaded | {token_limit:int, row_limit:int}      | Emitted when splitter loads config  |
| file_split                | {input_file:str, output_files:list}   | Emitted when a file is split        |
"""

DEFAULT_EXAMPLES_TEXT = "This file would contain examples of the target style. If you want it to be used in the prompt, add it to the config.yaml file."

DEFAULT_PROMPTS = {
    'batch_evaluation_prompt': (
        'You are an evaluator trying to determins the closeness of a response to a given style, examples of which will follow. Given the following examples, evaluate whether or not the response matches the target style.\n\n'
        'Examples:\n{dynamic_examples}\n\n'
        'Scoring should be as follows:\n'
        '5 - Perfect match\n'
        '4 - Very close\n'
        '3 - Somewhat close\n'
        '2 - Not close\n'
        '1 - No match\n\n'
        'Output only the numerical scores, one per line, in the same order as inputs.'
    ),
    'batch_evaluation_prompt_generic': (
        'You are an evaluator. Given the following message, rate its overall quality on a scale of 1 to 5.\n\n'
        'The scale is as follows:\n'
        '5 - Excellent\n'
        '4 - Good\n'
        '3 - Average\n'
        '2 - Poor\n'
        '1 - Very poor\n\n'
        'Output only the numerical score.'
    )
}

def ensure_config_files():
    '''
    Ensures dir structure is present and if not, creates a default one. 
    
    '''
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    input_dir = (CONFIG_DIR.parent / DEFAULT_CONFIG['input_dir']).resolve()
    output_dir = (CONFIG_DIR.parent / DEFAULT_CONFIG['output_dir']).resolve()
    examples_dir = (CONFIG_DIR.parent / DEFAULT_CONFIG['examples_dir']).resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.parent.mkdir(parents=True, exist_ok=True)
    if not examples_dir.exists():
        with open(examples_dir, 'w', encoding='utf-8') as f:
            f.write(DEFAULT_EXAMPLES_TEXT)
    if not CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'w') as f:
            yaml.safe_dump(DEFAULT_CONFIG, f)
    if not PROMPTS_PATH.exists():
        with open(PROMPTS_PATH, 'w') as f:
            yaml.safe_dump(DEFAULT_PROMPTS, f)

def is_examples_file_default(examples_path):
    """
    Checks if the examples file contains only the default text (i.e., not customized by the user).
    """
    try:
        with open(examples_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        return content == DEFAULT_EXAMPLES_TEXT
    except Exception:
        return True 

def load_config(config_path=None):
    """
    Loads the batch grading configuration from the specified config YAML file.
    If config_path is None, loads from config/config.yaml.
    Auto-creates config, prompts, & examples files with defaults if missing.
    Prefers environment variable for API key since that's secure or some shit.
    Returns:
        dict: Configuration parameters.
    Raises RuntimeError if it encounters a badly formatted config file.
    Raises ValueError if it encounters a missing config file.
    """
    if config_path is None:
        config_path = CONFIG_PATH
    else:
        config_path = Path(config_path)
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
            config = DEFAULT_CONFIG.copy()
        else:
            merged = DEFAULT_CONFIG.copy()
            merged.update(config)
            config = merged
    except Exception as e:
        raise RuntimeError(f"Error loading config file {config_path}: {e}")
    env_api_key = os.getenv('OPENAI_API_KEY')
    if env_api_key:
        config['openai_api_key'] = env_api_key
    return config 