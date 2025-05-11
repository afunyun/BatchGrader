# BatchGrader Application Flow Diagram

This document provides a visual representation of the BatchGrader application's architecture and flow as of v0.2.6-pre

## Component Diagram

```mermaid
%%{init: { 'theme': 'dark', 'themeVariables': {
      'background': '#0d1117',
      'primaryColor': '#21262d',      /* Background of the class boxes */
      'primaryTextColor': '#c9d1d9', /* Text color inside class boxes */
      'primaryBorderColor': '#8b949e',/* Border of the class boxes */
      'lineColor': '#8b949e',        /* Arrow and line color */
      'textColor': '#c9d1d9'         /* General text color (e.g., for 'uses' labels) */
}}}%%
classDiagram
    class batch_runner {
        +process_file(filepath)
        +get_request_mode(args)
        +print_token_cost_stats()
        +print_token_cost_summary(summary)
        +resolve_and_load_input_file(file_arg)
        +get_token_counter(system_prompt_content, response_field, enc)
        +main()
    }
    
    class config_loader {
        +DEFAULT_CONFIG
        +DEFAULT_PROMPTS
        +ensure_config_files()
        +is_examples_file_default(examples_path)
        +load_config()
    }
    
    class data_loader {
        +load_data(filepath)
        +save_data(df, filepath)
    }
    
    class evaluator {
        +load_prompt_template(name)
    }
    
    class input_splitter {
        +split_file_by_token_limit(input_path, token_limit, count_tokens_fn, response_field, row_limit, output_dir, file_prefix)
    }
    
    class llm_client {
        -api_key
        -model
        -endpoint
        -max_tokens
        -poll_interval
        -client
        +__init__(model, api_key, endpoint)
        -_prepare_batch_requests(df, system_prompt_content, response_field_name)
        -_upload_batch_input_file(requests_data, base_filename_for_tagging)
        -_manage_batch_job(input_file_id, source_filename)
        -_process_batch_outputs(batch_job_obj, df_with_custom_ids)
        +run_batch_job(df, system_prompt_content, response_field_name, base_filename_for_tagging)
    }
    
    class token_tracker {
        +update_token_log(api_key, tokens_submitted, date_str)
        +log_token_usage_event(api_key, model, input_tokens, output_tokens, timestamp, request_id)
        +load_token_usage_events()
        +get_token_usage_summary(start_date, end_date, group_by)
        +get_total_cost(start_date, end_date)
        +get_token_usage_for_day(api_key, date_str)
    }
    
    class CostEstimator {
        -_pricing
        -_csv_path
        -_load_pricing()
        +estimate_cost(model, n_input_tokens, n_output_tokens)
    }
    
    batch_runner --> config_loader : uses
    batch_runner --> data_loader : uses
    batch_runner --> evaluator : uses
    batch_runner --> input_splitter : uses
    batch_runner --> llm_client : uses
    batch_runner --> token_tracker : uses
    batch_runner --> CostEstimator : uses
    
    llm_client --> config_loader : uses
    token_tracker --> CostEstimator : uses pricing data
    input_splitter --> config_loader : uses
    evaluator --> config_loader : uses
```

## Sequence Diagram (Main Flow)

```mermaid
%%{init: { 'theme': 'dark' } }%%
sequenceDiagram
    participant User
    participant batch_runner
    participant config_loader
    participant data_loader
    participant evaluator
    participant llm_client
    participant input_splitter
    participant token_tracker
    participant CostEstimator
    
    User->>batch_runner: Run with arguments
    batch_runner->>config_loader: load_config()
    config_loader-->>batch_runner: config
    
    alt --count-tokens flag
        batch_runner->>data_loader: load_data(filepath)
        data_loader-->>batch_runner: dataframe
        batch_runner->>evaluator: load_prompt_template()
        evaluator-->>batch_runner: prompt_template
        batch_runner->>batch_runner: Count tokens
        batch_runner-->>User: Display token counts
    else --split-tokens flag
        batch_runner->>data_loader: load_data(filepath)
        data_loader-->>batch_runner: dataframe
        batch_runner->>evaluator: load_prompt_template()
        evaluator-->>batch_runner: prompt_template
        batch_runner->>batch_runner: Count tokens
        batch_runner->>input_splitter: split_file_by_token_limit()
        input_splitter-->>batch_runner: output_files, token_counts
        batch_runner-->>User: Display split results
    else normal batch processing
        batch_runner->>batch_runner: process_file(filepath)
        activate batch_runner
        batch_runner->>data_loader: load_data(filepath)
        data_loader-->>batch_runner: dataframe
        batch_runner->>evaluator: load_prompt_template()
        evaluator-->>batch_runner: prompt_template
        batch_runner->>batch_runner: Count tokens
        batch_runner->>token_tracker: update_token_log()
        batch_runner->>llm_client: run_batch_job()
        activate llm_client
        llm_client->>llm_client: _prepare_batch_requests()
        llm_client->>llm_client: _upload_batch_input_file()
        llm_client->>llm_client: _manage_batch_job()
        llm_client->>llm_client: _process_batch_outputs()
        llm_client-->>batch_runner: dataframe with results
        deactivate llm_client
        batch_runner->>data_loader: save_data()
        batch_runner->>CostEstimator: estimate_cost()
        CostEstimator-->>batch_runner: cost
        deactivate batch_runner
        batch_runner->>token_tracker: print_token_cost_stats()
        token_tracker->>token_tracker: get_token_usage_summary()
        token_tracker-->>batch_runner: summary
        batch_runner-->>User: Display results and stats
    end
```

## Data Flow Diagram

```mermaid
%%{init: { 'theme': 'dark' } }%%
flowchart TD
    A[Input Files] --> B[batch_runner.py]
    B --> C{Mode?}
    C -->|Count Tokens| D[Count and display token usage]
    C -->|Split Tokens| E[Split files by token limit]
    C -->|Batch Processing| F[Process files with OpenAI API]
    
    F --> G[Load Data]
    G --> H[Prepare Prompts]
    H --> I[Count Tokens]
    I --> J[Check Token Limits]
    J --> K[Submit Batch Job]
    K --> L[Process Results]
    L --> M[Save Results]
    M --> N[Estimate Cost]
    N --> O[Display Stats]
    
    P[config.yaml] --> B
    Q[prompts.yaml] --> H
    R[examples.txt] --> H
    S[pricing.csv] --> N
    
    E --> T[Split Output Files]
    T --> A
    
    M --> U[Output Files]
    O --> V[Token Usage Log]
```

## Component Descriptions

1. **batch_runner.py**: The main entry point of the application. Handles CLI arguments, processes input files, and orchestrates the workflow.

2. **config_loader.py**: Loads configuration from config.yaml and provides default values.

3. **data_loader.py**: Handles loading and saving data in different formats (CSV, JSON, JSONL).

4. **evaluator.py**: Loads prompt templates from prompts.yaml.

5. **input_splitter.py**: Splits input files into parts that do not exceed a specified token or row limit.

6. **llm_client.py**: Interacts with the OpenAI Batch API to submit batch jobs and process results.

7. **token_tracker.py**: Tracks and aggregates OpenAI API token usage for both API limit enforcement and historical/cost tracking.

8. **cost_estimator.py**: Estimates API costs based on the pricing data in docs/pricing.csv.

## Configuration Files

1. **config.yaml**: Contains configuration parameters like input/output directories, model name, token limits, etc.

2. **prompts.yaml**: Contains prompt templates for evaluation.

3. **examples.txt**: Contains examples of the target style for contextual evaluation.

4. **pricing.csv**: Contains pricing data for different OpenAI models.

## Workflow Modes

1. **Count Tokens**: Counts tokens in input files and displays statistics.

2. **Split Tokens**: Splits input files into parts that do not exceed the configured token limit.

3. **Batch Processing**: Processes input files using the OpenAI Batch API, saves results, and displays statistics.