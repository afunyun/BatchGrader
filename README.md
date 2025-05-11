# BatchGrader: Batch LLM Evaluation with OpenAI Batch API

## Overview

BatchGrader is a quick and dirty tool for batch evaluation of text data using the OpenAI Batch API. It is designed for grading, scoring, or classifying large sets of LLM-generated responses or any text data.

All configuration is managed via simple YAML.

## Features

- **YAML Configuration:** All runtime settings are in `config/config.yaml`.
- **Prompt Templates:** Prompts are managed in `config/prompts.yaml`.
- **Supports examples:** You can provide examples to the LLM in case your evaluation criteria require this, allowing the evaluator LLM additional context without having to directly change your prompt itself.
- **Supports CSV, JSON, JSONL:** Flexible input/output formats.
- **Cost Estimation:** BatchGrader can estimate the cost of running a batch job based on the number of tokens used. This is done using the OpenAI Batch API pricing. This is located in 'docs/pricing.csv', and it isn't dynamically updated lol. Make sure it's accurate.  

---

### 🚧 Current TODO / Action Items

- **Support richer evaluation outputs:**  
  Honestly have only tried this with the format of a single number rating. Should allow free-form or multi-field LLM responses, and ensure downstream code can handle these without exploding.
  (Probably already works tbh, just wary of saying you can do this without actually trying it.)

- **Improve tiktoken handling:**  
  Convert the tiktoken dependency check into a config option, or relocate it to the main execution loop.  

---

## Setup & Installation

**I am on Windows 11. I have not tested this on linux, but I have no reason to believe it wouldn't work just fine. You may need to modify the setup for bash rather than powershell.**

1. **Prerequisites:**
   - Python 3.7+
   - recommended:
   - `astral-uv` (rather than vanilla pip)
   - else:
   - `pip`

2. **virtual env, recommended**
    - git clone the repo
    - As is tradition, activate a venv just in case you have conflicting packages. I recommend astral-uv, always:

    ```powershell
    uv venv batchvenv
    batchvenv\Scripts\activate
    ```

    - If you don't do this, 3-4 people (random, selected from global population) suddenly lose 3 mm of length from every hair of their body. Don't do that.

3. **Install Dependencies:**

   ```powershell
   uv pip install -r requirements.txt
   ```

4. **Configure the System:**
   - Edit `config/config.yaml` to set your OpenAI API key and other parameters (see below).
   - Edit `config/prompts.yaml` to customize the evaluation prompt if needed.

## Configuration (YAML)

**On first run, system "should" create defaults: /config and /input. It will generate a default config.yaml and prompts.yaml, customize as needed:**

All runtime configuration is in `config/config.yaml`

Defaults:

```yaml
input_dir: input            # Directory for input files
output_dir: output          # Directory for output files
examples_dir: examples/examples.txt
openai_model_name: gpt-4o-mini-2024-07-18  # OpenAI model to use
openai_api_key: YOUR_OPENAI_API_KEY_HERE   # Your OpenAI API key
poll_interval_seconds: 60      # Polling interval for batch job status (seconds)
max_tokens_per_response: 1000  # Max tokens per LLM response. Not really super applicable unless you're expecting a response that isn't a single number.
response_field: response       # Field/column in input data to evaluate
batch_api_endpoint: /v1/chat/completions   # OpenAI batch API endpoint... not usually something to change

```

## Prompt Configuration

Prompts are managed in `config/prompts.yaml`. The system generated defaults as follows are used, depending on whether or not an examples file is provided (it will only do the examples version if the text is different than the default system created examples.txt):

```yaml
batch_evaluation_prompt_generic: |
    You are an evaluator. Given the following message, rate its overall quality on a scale of 1 to 5.
    The scale is as follows:
    5 - Excellent
    4 - Good
    3 - Average
    2 - Poor
    1 - Very poor
    Output only the numerical score.
```

```yaml
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
```

Edit this prompt to match your evaluation criteria. The prompt should instruct the LLM to output only the required result (e.g., a number per line). The defaults are obviously quite lacking, they are not really intended for use other than to be 'functional' (to not cause an error) and an example of what you might want to include. You should elaborate as much as you want, the better your prompt, the better your result.

## Usage

1. **Prepare Input Data:**
   - Place your input files (CSV, JSON, or JSONL) in the `input/` directory.
   - Ensure your files contain the field specified by `response_field` in `config.yaml`.
   - Optionally, if you need to provide examples in the prompts, you may add examples

2. **Run the Batch Grader:**

   ```bash
   cd src
   python batch_runner.py
   ```

   - The runner will process each file in the input directory, submit a batch job, and save results in the output directory once received. It will check for a completed job based on the specified value in the config. See notes below

3. **Check Output:**
   - Output files will have the same format as input, with an added `llm_score` column containing the evaluation result.
   - Errors or partial results will be saved with an `ERROR_` prefix in the output directory.

```markdown
    Rate limits
    -----------

    Batch API rate limits are separate from existing per-model rate limits. The Batch API has two new types of rate limits:

    1.  **Per-batch limits:** A single batch may include up to 50,000 requests, and a batch input file can be up to 200 MB in size. Note that `/v1/embeddings` batches are also restricted to a maximum of 50,000 embedding inputs across all requests in the batch.
    2.  **Enqueued prompt tokens per model:** Each model has a maximum number of enqueued prompt tokens allowed for batch processing. You can find these limits on the [Platform Settings page](/settings/organization/limits).

    There are no limits for output tokens or number of submitted requests for the Batch API today. Because Batch API rate limits are a new, separate pool, using the Batch API will not consume tokens from your standard per-model rate limits, thereby offering you a convenient way to increase the number of requests and processed tokens you can use when querying our API.
```

**OpenAI themselves limit the amount of requests per batch. See above if you're wondering why there's a 50k/request cap.**

## Input/Output Formats

- **CSV:** Must have a header row. The column specified by `response_field` will be used.
- **JSON:** An array of objects. Each object should have the key specified by `response_field`.
- **JSONL:** Each line is a JSON object with the key specified by `response_field`.

## Troubleshooting & Notes

- **API Key:**  
  Ensure your OpenAI API key is set in your environment. IF YOU REALLY WANT TO, and I don't recommend it, you can set it in `config/config.yaml` under `openai_api_key: YOUR_OPENAI_API_KEY_HERE`

- **Model Availability:**  
  Use a model compatible with the OpenAI Batch API.
  See table below for available models and their pricing.
  
- **Batch API Turnaround:**  
  Batch jobs may take up to 24 hours, but are often faster.

- **Prompt Engineering:**  
  The quality of results depends on your prompt. Ensure it is clear and outputs only the required result.

- **Large Files:**  
  Each input file is submitted as a single batch job. The Batch API has limits on file size and number of requests per batch. See the snippet from OpenAI's docs above. I also included the full API reference page in the repo if you're curious.

- **[OpenAI documentation](https://platform.openai.com/docs/guides/batch)**  
  The docs snippet and the one in this repo are obviously static. I pulled them from the site on 5/10/2025; if it's any time after that, it may be horribly out of date. I recommend checking the link above.

**As of 5/10/2025, the following models are available and priced thusly in the Batch API:**
  
| Model                           | Input per 1M tokens | Output per 1M tokens |
| ------------------------------- | ------------------- | -------------------- |
| gpt-4.1-mini-2025-04-14         | $0.20               | $0.80                |
| gpt-4.1-nano-2025-04-14         | $0.05               | $0.20                |
| gpt-4.5-preview-2025-02-27      | $37.50              | $75.00               |
| gpt-4o-2024-08-06               | $1.25               | $5.00                |
| gpt-4o-mini-2024-07-18          | $0.08               | $0.30                |
| o1-2024-12-17                   | $7.50               | $30.00               |
| o1-pro-2025-03-19               | $75.00              | $300.00              |
| o3-2025-04-16                   | $5.00               | $20.00               |
| o4-mini-2025-04-16              | $0.55               | $2.20                |
| o3-mini-2025-01-31              | $0.55               | $2.20                |
| o1-mini-2024-09-12              | $0.55               | $2.20                |
| computer-use-preview-2025-03-11 | $1.50               | $6.00                |

---

## DISCLAIMER

**I may have royally messed something up as is tradition, so let me know if it fails catastrophically.**

## Directory & Execution Structure

```file
BatchGrader/
├── config/
│   ├── config.yaml         # All runtime configuration (YAML)
│   └── prompts.yaml        # Prompt templates (YAML)
├── examples/
│   ├── examples.txt        # If evaluating with criteria that require the LLM to see examples of a correct response, place here.
├── input/                  # Place your input data files here
├── output/                 # Processed files with scores will be saved here
├── src/
│   ├── batch_runner.py     # Main batch processing script
│   ├── llm_client.py       # Modular LLM/batch API client
│   ├── data_loader.py      # Data loading/saving utilities
│   ├── evaluator.py        # Prompt loading utilities
│   ├── config_loader.py    # YAML config loader
│   └── cost_estimator.py   # Cost estimation for batch jobs
├── README.md               # This file
└── .gitignore
```

## Last updated: 2025-05-10 (added cost_estimator.py)
