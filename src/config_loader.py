import yaml
from pathlib import Path
import os

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'
CONFIG_PATH = CONFIG_DIR / 'config.yaml'
PROMPTS_PATH = CONFIG_DIR / 'prompts.yaml'

DEFAULT_CONFIG = {
    'input_dir': '../input',
    'output_dir': '../output',
    'examples_dir': '../examples/examples.txt',
    'openai_model_name': 'gpt-4o-mini-2024-07-18',
    # only use if you hate security (like i did in the first versions lol)
    # system will pull from environment variables FIRST if it's set there.
    # openai_api_key: YOUR_OPENAI_API_KEY_HERE
    'poll_interval_seconds': 60,
    'max_tokens_per_response': 1000,
    'response_field': 'response',
    'batch_api_endpoint': '/v1/chat/completions',
    'token_limit': 2_000_000
}

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

def load_config():
    """
    Loads the batch grading configuration from config/config.yaml.
    Auto-creates config, prompts, & examples files with defaults if missing.
    Prefers environment variable for API key since that's secure or some shit.
    Returns:
        dict: Configuration parameters.
    Raises RuntimeError if it encounters a badly formatted config file.
    Raises ValueError if it encounters a missing config file.
    """
    ensure_config_files()
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RuntimeError(f"Incorrectly formatted config.yaml: {e}") from e
    if not config:
        raise ValueError("Failed to load config file. How? Very carefully, since if it weren't there it should've been created.")
    env_api_key = os.getenv('OPENAI_API_KEY')
    if env_api_key:
        config['openai_api_key'] = env_api_key
    return config 