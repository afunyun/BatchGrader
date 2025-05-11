# **<div align="center"><b>Execution Flow ((Sorry, I like graphs))</b></div>**

```mermaid
graph TD

    subgraph 400["BatchGrader System"]
        subgraph 401["LLM Interaction"]
            416["LLM Interface (Python, OpenAI API)"]
        end
        subgraph 402["Core Processing Logic"]
            413["Batch Runner (Python)"]
            414["Prompt Processor (Python, YAML)"]
            415["Data I/O Handler (Python, Pandas)"]
        end
        subgraph 403["Configuration Management"]
            412["Configuration Loader (Python, YAML)"]
        end
    end
    subgraph 404["External Systems"]
        405["User (External Actor)"]
        406["File System (Storage for Data & Config)"]
        407["OpenAI Service (External LLM Provider)"]
    end
    %% Edges at this level (grouped by source)
    412["Configuration Loader (Python, YAML)"] -->|Reads configuration files from| 406["File System (Storage for Data & Config)"]
    405["User (External Actor)"] -->|Initiates batch grading job| 413["Batch Runner (Python)"]
    414["Prompt Processor (Python, YAML)"] -->|Loads prompt templates from| 406["File System (Storage for Data & Config)"]
    415["Data I/O Handler (Python, Pandas)"] -->|Reads input data from / Writes results to| 406["File System (Storage for Data & Config)"]
    416["LLM Interface (Python, OpenAI API)"] -->|Makes API calls to| 407["OpenAI Service (External LLM Provider)"]
```

```mermaid
graph TD
    %% Main components
    BatchRunner["Batch Runner(batch_runner.py)"]
    LLMClient["LLM Client(llm_client.py)"]
    ConfigLoader["Config Loader(config_loader.py)"]
    DataIO["Data I/O Handler(data_loader.py)"]
    Evaluator["Evaluator(evaluator.py)"]
    CostEstimator["Cost Estimator(cost_estimator.py)"]
    
    %% External components
    User["UserExternal Actor"]
    FileSystem["File SystemCSV, JSON, JSONL files"]
    OpenAI["OpenAI Batch APIExternal Service"]
    
    %% Configuration files
    ConfigYAML["config.yamlRuntime settings"]
    PromptsYAML["prompts.yamlPrompt templates"]
    Examples["examples.txtExample texts"]
    Pricing["pricing.csvModel pricing data"]
    
    %% Relationships - User interactions
    User -->|Initiates batch job| BatchRunner
    
    %% Relationships - BatchRunner with other components
    BatchRunner -->|Uses| LLMClient
    BatchRunner -->|Uses| ConfigLoader
    BatchRunner -->|Uses| DataIO
    BatchRunner -->|Uses| Evaluator
    BatchRunner -->|Uses| CostEstimator
    
    %% Relationships - Config management
    ConfigLoader -->|Loads| ConfigYAML
    ConfigLoader -->|Loads| PromptsYAML
    Evaluator -->|Reads templates from| PromptsYAML
    BatchRunner -->|Reads examples from| Examples
    CostEstimator -->|Reads pricing from| Pricing
    
    %% Relationships - External systems
    LLMClient -->|Sends batch requests to| OpenAI
    OpenAI -->|Returns results to| LLMClient
    DataIO -->|Reads input from| FileSystem
    DataIO -->|Writes output to| FileSystem
    
    %% Visual grouping
    subgraph BatchGraderCore["BatchGrader Core System"]
        BatchRunner
        LLMClient
        ConfigLoader
        DataIO
        Evaluator
        CostEstimator
    end
    
    subgraph ExternalComponents["External Components"]
        User
        FileSystem
        OpenAI
    end
    
    subgraph ConfigurationFiles["Configuration Files"]
        ConfigYAML
        PromptsYAML
        Examples
        Pricing
    end
```

---

**<div align="center">Architecture Diagram (Combined Structure & Execution)</div>**

```mermaid
%% BatchGrader System Architecture & Execution Flow (Function-Level, Updated)
flowchart TD
  %% External Systems
  subgraph External_Systems["External Systems"]
    OpenAI_API["OpenAI API<br>External LLM Service"]
  end

  %% File System
  subgraph File_System["File System Storage"]
    FS_Config["config.yaml"]
    FS_Prompts["prompts.yaml"]
    FS_Input["Input Data Files (.csv, .json, .jsonl)"]
    FS_Output["Output Result Files (incl. ERROR_*)"]
    FS_Examples["User Example Data<br>(e.g., from 'examples_dir')"]
    FS_Pricing["docs/pricing.csv"]
  end

  %% Application Components
  subgraph BatchGraderApp["Batch Grader Application"]
    Comp_BatchRunner["Main Runner<br>(batch_runner.py)"]
    Comp_ConfigLoader["Configuration Loader<br>(config_loader.py)"]
    Comp_DataLoader["Data Loader<br>(data_loader.py)"]
    Comp_Evaluator["Prompt Evaluator<br>(evaluator.py)"]
    Comp_LLMClient["LLM Client<br>(llm_client.py)"]
    Comp_CostEstimator["Cost Estimator<br>(cost_estimator.py)"]
  end

  %% LLM Operations (detailed)
  subgraph LLM_Ops["LLM Batch Job (LLMClient.run_batch_job)"]
    D1["_prepare_batch_requests(df, prompt, response_field)"]
    D2["_upload_batch_input_file(requests, filename)"]
    D3["_manage_batch_job(input_file_id, filename)"]
    D4["_collect_batch_outputs(batch_job, df)"]
  end

  %% Main Execution Flow
  Comp_BatchRunner -- "load_config()" --> Comp_ConfigLoader
  Comp_ConfigLoader <-- "reads" --> FS_Config
  Comp_BatchRunner -- "load_data()" --> Comp_DataLoader
  Comp_DataLoader -- "reads" --> FS_Input
  Comp_BatchRunner -- "prepare prompt" --> Comp_Evaluator
  Comp_Evaluator -- "reads" --> FS_Prompts
  Comp_BatchRunner -- "reads content from" --> FS_Examples
  Comp_BatchRunner -- "instantiate & use" --> Comp_LLMClient
  Comp_BatchRunner -- "estimate costs" --> Comp_CostEstimator
  Comp_CostEstimator -- "reads" --> FS_Pricing
  Comp_LLMClient -- "run_batch_job()" --> D1
  D1 --> D2
  D2 -- "upload file" --> OpenAI_API
  D2 --> D3
  D3 -- "manage job, poll" --> OpenAI_API
  D3 --> D4
  D4 -- "get results" --> OpenAI_API
  D4 -- "returns results" --> Comp_LLMClient
  Comp_LLMClient -- "returns outputss" --> Comp_BatchRunner
  Comp_BatchRunner -- "save_data()" --> Comp_DataLoader
  Comp_DataLoader -- "writes" --> FS_Output
  Comp_BatchRunner -- "on error, save error file" --> FS_Output

  %% Config Files: Read-Only Relationship
  FS_Config <-- "read (not written except at project init)" --> Comp_BatchRunner
  FS_Prompts <-- "read (not written except at project init)" --> Comp_Evaluator
  FS_Examples <-- "read (not written except at project init; applies to all files in examples/)" --> Comp_BatchRunner
  FS_Input <-- "read (input directory may be created if missing, but files are only read)" --> Comp_BatchRunner

  %% Control & Data Flow
  Comp_BatchRunner -- "main: iterates input files, calls process_file()" --> Comp_BatchRunner
  Comp_BatchRunner -- "uses config values" --> Comp_ConfigLoader
  Comp_BatchRunner -- "loads examples" --> Comp_Evaluator

  %% Styling
  style External_Systems fill:#161b22,stroke:#795548,stroke-width:1.5px
  style File_System fill:#161b22,stroke:#2e7d32,stroke-width:1.5px
  style BatchGraderApp fill:#161b22,stroke:#1e88e5,stroke-width:1.5px
  style LLM_Ops stroke-width:1px
```

---

**<div align="center">Sequence Diagram (File Processing Flow)</div>**

```mermaid
sequenceDiagram
    participant User as User/System
    participant Runner as batch_runner.py
    participant ConfigL as config_loader.py
    participant DataL as data_loader.py
    participant Evaluator as evaluator.py
    participant LLMCli as llm_client.py
    participant CostEst as cost_estimator.py
    participant OpenAI as OpenAI API

    User->>+Runner: main() / Starts execution
    Runner->>+ConfigL: load_config()
    ConfigL-->>-Runner: config object
    Runner->>Runner: Identifies input files in INPUT_DIR
    loop For each input file
        Runner->>Runner: process_file(filepath)
        Runner->>+DataL: load_data(filepath)
        DataL-->>-Runner: DataFrame (df)
        alt df is empty
            Runner->>Runner: Log "No data loaded" and skip
        else df is not empty
            Runner->>ConfigL: is_examples_file_default(examples_path)
            ConfigL-->>Runner: boolean (is_default)
            alt is_examples_file_default is false
                Runner->>Runner: Read examples file content
                Runner->>+Evaluator: load_prompt_template("batch_evaluation_prompt")
                Evaluator-->>-Runner: prompt_template
                Runner->>Runner: Format system_prompt_content with examples
            else is_examples_file_default is true
                Runner->>+Evaluator: load_prompt_template("batch_evaluation_prompt_generic")
                Evaluator-->>-Runner: system_prompt_content
            end

            Runner->>+LLMCli: Instantiate LLMClient()
            LLMCli-->>-Runner: llm_client instance
            Runner->>+LLMCli: run_batch_job(df, system_prompt_content, response_field, filename)
            Note over LLMCli,OpenAI: _prepare_batch_requests()\n_upload_batch_input_file() -> OpenAI\n_manage_batch_job() -> polls OpenAI\n_collect_batch_outputs() <- from OpenAI
            LLMCli-->>OpenAI: Batch API calls (upload, create, poll, retrieve)
            OpenAI-->>LLMCli: Batch job results
            LLMCli-->>-Runner: DataFrame with results (df_with_results)

            Runner->>+DataL: save_data(df_with_results, output_path)
            DataL-->>-Runner: status
            Runner->>Runner: Log success and stats

            alt Cost Estimation
                Runner->>Runner: Check tiktoken, count input/output tokens in df_with_results
                Runner->>+CostEst: CostEstimator.estimate_cost(model, input_tokens, output_tokens)
                CostEst-->>-Runner: estimated_cost
                Runner->>Runner: Log estimated cost
            end
        end
    end
    Runner-->>-User: Processing finished
```
