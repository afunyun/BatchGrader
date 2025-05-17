# BatchGrader: Batch LLM Evaluation with OpenAI Batch API

## Recent Changes

- **Version 0.6.1.0** (2025-05-17): Implemented reprocessing for failed items and automatic API call retries. See `docs/CHANGELOG.md` for details.
- **Version 0.6.0.3** (2025-05-17): Major release! Core batch processing and evaluation functionality is robust, with 86%+ test coverage. Ready for broader use. All major features work and the system is no longer "embarrassing" for public use.
- **Version 0.5.9** (2025-05-16): Completed test coverage improvements across multiple modules. 100% coverage for several important modules, and near-perfect coverage for several other modules. All tests now running with pytest.

## Overview

BatchGrader is a quick and dirty tool for batch evaluation of text data using the OpenAI Batch API. It is designed for grading, scoring, or classifying large sets of LLM-generated responses or any text data.

All configuration is managed via simple YAML.

## Features

- YAML-based configuration (config/config.yaml, config/prompts.yaml).
- Contextual evaluation with user-provided examples.
- Supports CSV, JSON, JSONL input/output.
- Cost estimation via docs/pricing.csv.
- Token tracking and input splitting utilities.

---

### 🚧 Current TODO / Action Items

- **Support richer evaluation outputs:**
  **Unofficially implemented by accident**
  It is fully possible to run the grader and process outputs of arbitrary length, I just personally needed a numerical rating. It should handle it without issue.

- **True Async Processing:**
  **IN PROGRESS**
  Allow for using the terminal interface while job is running - check status etc

---

## Setup & Installation

**I am on Windows 11. I have not tested this on linux, but I have no reason to believe it wouldn't work just fine. You may need to modify the setup for bash rather than powershell.**

1. **Prerequisites**

   - Python 3.7+
   - Recommended: `uv` (preferred for `requirements.txt`) or `pip`.

2. **Virtual Environment (Recommended)**

   - Clone the repo:

     ```powershell
     git clone https://github.com/afunyun/BatchGrader.git
     cd BatchGrader
     ```

   - "Optional but not running in a venv is asking for a world of hurt, promise" Sync and activate the virtual environment (exact script may vary based on your system):

     ```powershell
     uv sync
     .venv\Scripts\activate.ps1
     ```

     If you skip this step, 3-4 random people (selected globally) will lose 3 mm of length f
     rom every hair on their body. Please don't do that.

   - **Dependency Management**

     - The project uses `pyproject.toml` with Poetry for dependency management.
     - For development, install with:

       ```powershell
       # Install in development mode with all dependencies
       pip install -e ".[dev]"
       
       # Or using uv (faster):
       uv pip install -e "."
       ```

     - For production, use `requirements.txt`:

       ```powershell
       uv pip install -r requirements.txt
       ```

3. **Usage**

   - Always make sure your `.venv` is active—if it's not, something _will_ break, almost guaranteed.
   - Rename `config/config.yaml.example` to `config/config.yaml` and fill in the values.
     (If you get a config loading error, check that these files are in the expected `/config` location.)
   - Then run:

     ```powershell
     python -m src.batch_runner [args]
     ```

This of course is just to run the batch runner with nothing else. See below for more advanced usage.

## Dependencies

    - `rich`: For live-updating CLI progress/status tables and colorized output.
    - `pandas`: For reading/writing CSV/JSON/JSONL files.
    - `openai`: For interacting with the OpenAI API.
    - `python-dotenv`: For loading environment variables from a .env file.
    - `tiktoken`: For counting tokens in text.
    - `pyyaml`: For loading YAML files.
    - Various others in requirements.txt and pyproject.toml - these are the main breaking ones however

4. **Configuration:**

- Edit config/config.yaml (API key, paths) and config/prompts.yaml (evaluation prompts).

## Configuration (YAML)

**Default config/ and input/ directories, along with config.yaml and prompts.yaml, are created on first run if they don't exist. Customize as needed.**

All runtime configuration is in `config/config.yaml`

Defaults:

```yaml
input_dir: input
output_dir: output
examples_dir: examples/examples.txt
openai_model_name: gpt-4o-mini-2024-07-18
# API key is pulled from environment variable OPENAI_API_KEY if set
# otherwise uncomment and set below:
# openai_api_key: YOUR_OPENAI_API_KEY_HERE
poll_interval_seconds: 60
max_tokens_per_response: 1000
response_field: response
batch_api_endpoint: /v1/chat/completions
token_limit: 2000000
split_token_limit: 500000
```

## Prompt Configuration

Prompts are managed in `config/prompts.yaml`. The system generated defaults as follows are used, depending on whether or not an examples file is provided (it will only do the examples version if the text is different than the default system created examples.txt):

<details>
<pre>
batch_evaluation_prompt_generic: |
    You are an evaluator. Given the following message, rate its overall quality on a scale of 1 to 5.
    The scale is as follows:
    5 - Excellent
    4 - Good
    3 - Average
    2 - Poor
    1 - Very poor
    Output only the numerical score.

<br>
batch_evaluation_prompt: |
    You are an evaluator trying to determins the closeness of a response to a given style, examples of which will follow. Given the following examples, evaluate whether or not the response matches the target style.
    Examples:\n{dynamic_examples}\n\n
    Scoring should be as follows:
    5 - Perfect match
    4 - Very close
    3 - Somewhat close
    2 - Not close
    1 - No match
    Output only the numerical scores, one per line, in the same order as inputs.
</pre>
</details>

Edit this prompt to match your evaluation criteria. The prompt should instruct the LLM to output only the required result (e.g., a number per line). The defaults are obviously quite lacking, they are not really intended for use other than to be 'functional' (to not cause an error) and an example of what you might want to include. You should elaborate as much as you want, the better your prompt, the better your result.

### Usage

#### Command-Line Usage

**05/11/2025 - added the following options for command line arguments, if you wish to count/split inputs before actually submitting them to API:**

```powershell
# Count tokens in all input files
python batch_runner.py --count-tokens

# Split input files into parts not exceeding the configured token limit
python batch_runner.py --split-tokens

# Only process a specific file
python batch_runner.py --count-tokens --file myfile.csv
python batch_runner.py --split-tokens --file myfile.csv

# Default batch processing (no arguments)
python batch_runner.py
```

- `--count-tokens`: Prints total, average, and max token counts per file (uses your prompt template and tiktoken).
- `--split-tokens`: Splits input files into chunks that do not exceed the configured token limit, outputting `{filename}_part1.csv`, `{filename}_part2.csv`, etc.
- `--file <filename>`: Restrict operation to a specific file in the input directory.

If neither `--count-tokens` nor `--split-tokens` is specified, the system runs the standard batch evaluation workflow.

---

1. **Prepare Input Data:**

   - Place your input files (CSV, JSON, or JSONL) in the `input/` directory.
   - Ensure your files contain the field specified by `response_field` in `config.yaml`.
   - Optionally, if you need to provide examples in the prompts, you may add examples.

2. **Run the Batch Grader:**

   - Make sure you are in the BatchGrader directory.

   ```bash
   python -m src.batch_runner [args]
   ```

   - The runner will process each file in the input directory, submit a batch job, and save results in the output directory once received. It will check for a completed job based on the specified value in the config. See notes below.

3. **Check Output:**
   - Results are in `output/` with an added `llm_score` column. Errors are prefixed with `ERROR_`.

#### Rate Limits

- The Batch API has separate rate limits (per-batch up to 50k requests/200MB file, and enqueued prompt tokens per model). It doesn't consume your standard API rate limits.
- Default token_limit in config.yaml is 2,000,000 TPD (Tier 1); adjust as needed based on your [OpenAI Organization Limits](https://platform.openai.com/settings/organization/limits).
- For full details, see [OpenAI's documentation](https://platform.openai.com/docs/guides/batch) or BATCH_API_REFERENCE.md. See notes regarding this, however.

If you so choose, you can increase/decrease this via the config.yaml:

```yaml
token_limit: 2_000_000 #change to whatever your limit is
```

#### Input/Output Formats

- **CSV:** Must have a header row. The column specified by `response_field` will be used.
- **JSON:** An array of objects. Each object should have the key specified by `response_field`.
- **JSONL:** Each line is a JSON object with the key specified by `response_field`.

#### Assorted Notes

- API Key: Set OPENAI_API_KEY environment variable (recommended) or in config.yaml.
- Model Availability: Use models compatible with Batch API (see pricing table or OpenAI docs).
- Batch API Turnaround: Up to 24 hours (often faster).
- Prompt Engineering: Clear prompts yield better results.
- Large Files: Submitted as single batch jobs; mind API limits (see Rate Limits or BATCH_API_REFERENCE.md).

  The OpenAI docs I've provided are obviously static. I pulled them from the site on 5/10/2025; if it's any time after that, it may be horribly out of date. I recommend checking the link above.

**Pricing as of 2025-05-10. Verify with OpenAI or docs/pricing.csv (update manually as needed).**

<details>  
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Input per 1M tokens</th>
      <th>Output per 1M tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>gpt-4.1-mini-2025-04-14</td>
      <td>$0.20</td>
      <td>$0.80</td>
    </tr>
    <tr>
      <td>gpt-4.1-nano-2025-04-14</td>
      <td>$0.05</td>
      <td>$0.20</td>
    </tr>
    <tr>
      <td>gpt-4.5-preview-2025-02-27</td>
      <td>$37.50</td>
      <td>$75.00</td>
    </tr>
    <tr>
      <td>gpt-4o-2024-08-06</td>
      <td>$1.25</td>
      <td>$5.00</td>
    </tr>
    <tr>
      <td>gpt-4o-mini-2024-07-18</td>
      <td>$0.08</td>
      <td>$0.30</td>
    </tr>
    <tr>
      <td>o1-2024-12-17</td>
      <td>$7.50</td>
      <td>$30.00</td>
    </tr>
    <tr>
      <td>o1-pro-2025-03-19</td>
      <td>$75.00</td>
      <td>$300.00</td>
    </tr>
    <tr>
      <td>o3-2025-04-16</td>
      <td>$5.00</td>
      <td>$20.00</td>
    </tr>
    <tr>
      <td>o4-mini-2025-04-16</td>
      <td>$0.55</td>
      <td>$2.20</td>
    </tr>
    <tr>
      <td>o3-mini-2025-01-31</td>
      <td>$0.55</td>
      <td>$2.20</td>
    </tr>
    <tr>
      <td>o1-mini-2024-09-12</td>
      <td>$0.55</td>
      <td>$2.20</td>
    </tr>
    <tr>
      <td>computer-use-preview-2025-03-11</td>
      <td>$1.50</td>
      <td>$6.00</td>
    </tr>
  </tbody>
</table>
</details>

---

## DISCLAIMER

**I wouldn't call myself a programming mastermind, so I may have royally messed something up. Let me know if it fails catastrophically.**

## Directory Structure

```text
BatchGrader/
├── config/
│   ├── config.yaml
│   ├── config.yaml.example
│   ├── prompts.yaml
│   └── prompts.yaml.example
├── docs/
│   ├── BATCH_API_REFERENCE.md
│   ├── CHANGELOG.md
│   ├── code_review.md
│   ├── codebase_best_practices.md
│   ├── diagramsLULE.md
│   ├── pricing.csv
│   ├── scratchpad.md
│   ├── testing_info.md
│   └── completed_reviews_old_docs/
├── examples/
│   ├── afunyun_examples.txt
│   └── examples.txt
├── input/
│   ├── .keep
│   ├── _chunked/                # Auto-generated chunked input files
│   └── afunyun_dataset.csv      # Example dataset
├── output/
│   ├── batch_results/           # Batch job results
│   ├── logs/                    # Persistent logs (.keep present)
│   ├── token_usage_events.jsonl # Token usage event log
│   └── token_usage_log.json     # Token usage summary
├── src/
│   ├── __init__.py
│   ├── batch_job.py
│   ├── batch_runner.py
│   ├── cli.py
│   ├── config_loader.py
│   ├── constants.py
│   ├── cost_estimator.py
│   ├── data_loader.py
│   ├── evaluator.py
│   ├── exceptions.py
│   ├── file_processor.py
│   ├── file_utils.py
│   ├── input_splitter.py
│   ├── llm_client.py
│   ├── llm_utils.py
│   ├── log_utils.py
│   ├── logger.py
│   ├── output/                  # Output helpers (if present)
│   ├── prompt_utils.py
│   ├── rich_display.py
│   ├── token_tracker.py
│   ├── token_utils.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   ├── config/
│   ├── conftest.py
│   ├── input/
│   ├── logs/
│   ├── output/
│   ├── run_all_tests.py
│   ├── small_utilities/
│   ├── test_batch_job.py
│   ├── test_batch_runner.py
│   ├── test_check_token_limits.py
│   ├── test_cli.py
│   ├── test_cli_additional.py
│   ├── test_config_loader.py
│   ├── test_constants.py
│   ├── test_cost_estimator.py
│   ├── test_data_loader.py
│   ├── test_evaluator.py
│   ├── test_exceptions.py
│   ├── test_file_processor.py
│   ├── test_file_processor_additional.py
│   ├── test_file_processor_common_paths.py
│   ├── test_file_utils.py
│   ├── test_helpers.py
│   ├── test_input_splitter.py
│   ├── test_llm_client.py
│   ├── test_llm_utils.py
│   ├── test_log_utils.py
│   ├── test_logger.py
│   ├── test_prompt_utils.py
│   ├── test_rich_display.py
│   ├── test_splitter.py
│   ├── test_token_tracker.py
│   ├── test_token_utils.py
│   └── test_utils.py
├── .gitignore
├── .markdownlint.json
├── .markdownlintignore
├── LICENSE
├── pyproject.toml
├── pytest.ini
├── README.md
├── release_tag.ps1
├── requirements.txt
├── uv.lock
```

- All chunked input files are auto-stored in `input/_chunked/`.
- All logs go to `output/logs/` (production) or `tests/logs/` (tests), with `.keep` files to ensure directory presence.
- Directory names are singular and standardized.

![architecture](docs/diagram_dark_bg.png)
