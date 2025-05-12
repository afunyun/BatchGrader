# BatchGrader: Batch LLM Evaluation with OpenAI Batch API

## Last updated: 2025-05-11

### [0.3.2] (2025-05-11)

- **Enforced required pricing.csv:** BatchGrader now fails fast if docs/pricing.csv is missing, raising FileNotFoundError. Pricing is now a hard requirement for all cost estimation and token tracking. Because we said so sucka.
- **Deep merge config enhancement:** User config files are now recursively merged with defaults, preventing accidental loss of nested settings

### [0.3.1] (2025-05-11)

- Fixed chunking (whoops)
- Fixed storage locations in general

### [0.3.0] (2025-05-11)

- Implemented rich library for console output rather than basic prints, should prevent massive console spamming and also just look better//be more usable/extensible
- Colorized text & status with emoji because why not
- Added summary table to show totals for jobs, successes, failures, errors, tokens, and cost.
- Persistent logging and event file improvements.

### [0.2.12] (2025-05-11)

- Fully integrated the [rich](https://rich.readthedocs.io/) library for a live-updating, colorized CLI job status table.
- Batch job polling and progress/status updates now appear in a single, clean table (no more repeated print statements).
- All job statuses, errors, and progress are visible at a glance; failures are highlighted.
- Logging remains robust: persistent logs are still written to file for debugging and traceability.
- Yeeted straggling print statements in llm_client.py
- See `docs/scratchpad.md` for rationale and implementation notes.

### [0.2.11] (2025-05-11)

- Added chunking subdir to prune extra files after processing

- only pruned once processing is complete - the full files contain the content so there is no information loss by doing so.
- Logger now always writes a clear startup message to every log file as soon as it is instantiated, guaranteeing that all log files contain at least one entry.
- LOG PRUNING/ARCHIVING
- To prevent unbounded growth of the output/logs/ directory, prune/rotate logs before logger instantiation
- Moves oldest logs to output/logs/archive/ if threshold exceeded, and deletes oldest archive logs if needed
- See docs/scratchpad.md for rationale and changelog
- This improves traceability and debugging, especially for CI/test runs or troubleshooting failed jobs.
- Logger always writes a startup message to avoid empty logs
- Logging colour coding + console output is working perfectly
- All error/exit paths now log to file before exiting w/ new logger
- Tests seem to be functioning, CLI args working properly, tests succeeding
- Directory structure has been repaired & is standard now.

### [0.2.10] (2025-05-11)

- Renamed `tests/test_inputs/` → `tests/input/` and `tests/test_outputs/` → `tests/output/`.
- Added `.keep` files to ensure important directories exist in both production and test environments.
- Updated all test configs and code references to match the new convention.
- Added `.gitignore` to reflect the new test output directory.
- Tests 3/5 are complete and functional. Rerun is in order due to changes and hopefully all 5 will run.

### [0.2.9] (2025-05-11)

- Added --config arg, stress tested --file cli arg.
- Created testing agenda and a testing script + multiple faulty configs/datasets for testing purposes.

### [0.2.8] (2025-05-11)

- Integration: Concurrent Processing in Main Workflow
  - Integrated concurrent batch processing into the main workflow (`process_file`).
  - Now, if forced chunking or token-based splitting is needed, `process_file` routes to `process_file_concurrently` for parallel execution.
  - Otherwise, it uses the legacy single-batch logic for backward compatibility.
- All results and errors are saved and logged as before.
- The system is now ready for targeted testing of both single-batch and concurrent batch modes.
- Implementation of the concurrent batch job system has begun.
- Config layer (config_loader.py, config.yaml) and chunk job abstraction (batch_job.py) are complete.
- The `_generate_chunk_job_objects` helper is implemented and tested for both forced and token-based chunking.

### [0.2.7] (2025-05-11)

> **Note:** Implementation of concurrent batch (multi-job) processing is underway! Soon, BatchGrader will be able to split large datasets into multiple jobs and process them in parallel for much greater speed and scalability. Version 0.3.0 should contain this, assuming everything doesn't explode.

### [0.2.6] (2025-05-11)

- Large update to `token_tracker.py` to allow for tracking token usage and cost over time

- Why? if no info i am sad. Also the prices are low enough that if you're not running massive batches you're essentially always seeing 0 for the cost - want to be able to track cost even over a large number of smaller runs. Also added additional methods to `batch_runner.py` to display cumulative statistics. Added CLI arguments for this functionality (toggle/override). Statistics will not appear on unsuccessful runs OR splitting/counting runs. Only successful executions with a response receive get added to the new log and trigger the new stats display. Updated error handling specifically for CLI args - if positioning is incorrect, has more useful error message.

### [0.2.5] (2025-05-11)

- Added `batch_runner.py` to allow for use of the command I provided XD

- Update pyproject version because it was a crisp 5 "versions" (updates) behind. Prep for update check method because I want to be able to check for updates. I change things at random and frequently like a boss.

### [0.2.4] (2025-05-11)

- Fixed tiktoken error output to actually raise an exception

- this is instead of printing an error to console and then exiting. Deleted redundant docs images (NOOOOO). Update console output to provide more information about the current runtime while the `batch_runner.py` is running - previously it was fairly hard to see what options were actually running. Added `split_token_limit` to config.yaml to allow for splitting input files into chunks that do not exceed the configured token limit. CLI and normal operation now respect this limit rather than the Token Limit if it is applicable.

### [0.2.3] (2025-05-11)

- Release fixes - everything seems to ...work... on my machine at least

- Testing leaves something to be desired. Also managed to annihilate the branches for a second.

### [0.2.2] (2025-05-11)

- Fix multiple input files continuing to run/submit jobs if a reply comes back with errors. (preliminary fix not fully tested)

### [0.2.1] (2025-05-11)

- fixed cli args which i managed to put in horribly broken, nice, unified loading logic as well for args

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
  **IN PROGRESS**
  Honestly have only tried this with the format of a single number rating. Should allow free-form or multi-field LLM responses, and ensure downstream code can handle these without exploding.
  (Probably already works tbh, just wary of saying you can do this without actually trying it.)

---

## Setup & Installation

**I am on Windows 11. I have not tested this on linux, but I have no reason to believe it wouldn't work just fine. You may need to modify the setup for bash rather than powershell.**

1. **Prerequisites:**
   - Python 3.7+
   - Recommended: uv (preferred for requirements.txt) or pip.

2. **Virtual Environment (Recommended)**

   - Clone the repo:

     ```powershell
     git clone https://github.com/afunyun/BatchGrader.git
     cd BatchGrader
     ```

   - (Optional but encouraged) Create and activate a virtual environment. I recommend astral-uv:

     ```powershell
     uv venv batchvenv
     batchvenv\Scripts\activate
     ```

     If you don't do this, 3-4 people (random, selected from global population) suddenly lose 3 mm of length from every hair of their body. Don't do that.

   - **Dependency Management (uv + requirements.txt is Canonical)**

     - Uses requirements.txt (managed with uv or pip). pyproject.toml is for compatibility only and not actively maintained.

     **Installing Dependencies:**

     ```powershell
     uv pip install -r requirements.txt
     ```

## Dependencies

    - `rich`: For live-updating CLI progress/status tables and colorized output.
    - `pandas`: For reading/writing CSV/JSON/JSONL files.
    - `openai`: For interacting with the OpenAI API.
    - `python-dotenv`: For loading environment variables from a .env file.
    - `tiktoken`: For counting tokens in text.
    - `pyyaml`: For loading YAML files.
    - Various others in requirements.txt and pyproject.toml - these are the main breaking ones however

3. **Configuration:**

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

## Usage

### Command-Line Usage

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
   - Optionally, if you need to provide examples in the prompts, you may add examples

2. **Run the Batch Grader:**

   ```powershell
   cd src
   python batch_runner.py
   ```

   - The runner will process each file in the input directory, submit a batch job, and save results in the output directory once received. It will check for a completed job based on the specified value in the config. See notes below

3. **Check Output:**
   - Results are in output/ with an added `llm_score` column. Errors are prefixed with `ERROR_`.

## Rate Limits

- The Batch API has separate rate limits (per-batch up to 50k requests/200MB file, and enqueued prompt tokens per model). It doesn't consume your standard API rate limits.
- Default token_limit in config.yaml is 2,000,000 TPD (Tier 1); adjust as needed based on your [OpenAI Organization Limits](https://platform.openai.com/settings/organization/limits).
- For full details, see [OpenAI's documentation](https://platform.openai.com/docs/guides/batch) or BATCH_API_REFERENCE.md. See notes regarding this, however.

If you so choose, you can increase/decrease this via the config.yaml:

```yaml
token_limit: 2_000_000 #change to whatever your limit is
```

## Input/Output Formats

- **CSV:** Must have a header row. The column specified by `response_field` will be used.
- **JSON:** An array of objects. Each object should have the key specified by `response_field`.
- **JSONL:** Each line is a JSON object with the key specified by `response_field`.

## Assorted Notes

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
│   └── prompts.yaml
├── docs/
│   ├── pricing.csv
│   └── scratchpad.md
├── examples/
│   └── examples.txt
├── input/
│   ├── _chunked/          # Auto-generated chunked input files (.keep for dir presence)
│   └── ... (your input files)
├── output/
│   ├── logs/              # Persistent logs (.keep present)
│   └── ... (results, token_usage_log.json)
├── tests/
│   ├── input/
│   ├── output/
│   └── logs/              # Test run logs (.keep present)
├── src/
│   ├── batch_runner.py         # Main entry point & CLI
│   ├── config_loader.py        # Loads config & defaults
│   ├── cost_estimator.py       # Cost estimation logic
│   ├── data_loader.py          # Reads/writes CSV/JSON/JSONL
│   ├── evaluator.py            # Prompt template mgmt
│   ├── input_splitter.py       # Utility for input splitting by token limit
│   ├── llm_client.py           # OpenAI Batch API client
│   ├── logger.py               # Modular logging utility
│   ├── log_utils.py            # Log pruning/archiving
│   ├── file_utils.py           # File/directory helpers (e.g., prune_chunked_dir)
│   ├── rich_display.py         # Rich CLI live tables
│   ├── token_tracker.py        # Tracks API token usage
│   └── __pycache__/
├── requirements.txt
├── pyproject.toml
├── README.md
└── ...
```

- All chunked input files are auto-stored in `input/_chunked/`.
- All logs go to `output/logs/` (production) or `tests/logs/` (tests), with `.keep` files to ensure directory presence.
- Directory names are singular and standardized.

![architecture](docs\diagram_dark_bg.png)
