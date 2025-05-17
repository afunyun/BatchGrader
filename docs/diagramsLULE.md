# BatchGrader Architecture Diagrams

## 1. High-Level System Architecture

### 1.1 Core System Architecture

```mermaid
graph TD
    subgraph "User Interface"
        CLI["CLI (src/cli.py)"]
    end

    subgraph "Core Logic / Orchestration"
        BatchRunner["Batch Runner (src/batch_runner.py)"]
        FileProcessor["File Processor (src/file_processor.py)"]
        BatchJobClass["BatchJob Class (src/batch_job.py)"]
    end

    subgraph "Data Management"
        DataLoader["Data Loader (src/data_loader.py)"]
        InputSplitter["Input Splitter (src/input_splitter.py)"]
    end

    subgraph "LLM API Interaction"
        LLMClient["LLM Client (src/llm_client.py)"]
        OpenAI_API["OpenAI Batch API (External)"]
    end

    subgraph "Configuration & Utilities"
        ConfigLoader["Config Loader (src/config_loader.py)"]
        Constants["Constants (src/constants.py)"]
        PromptUtils["Prompt Utils (src/prompt_utils.py)"]
        Evaluator["Evaluator (src/evaluator.py)"]
        TokenUtils["Token Utils (src/token_utils.py)"]
        TokenTracker["Token Tracker (src/token_tracker.py)"]
        CostEstimator["Cost Estimator (src/cost_estimator.py)"]
        LoggerUtil["Logger (src/logger.py, src/log_utils.py)"]
        FileUtils["File Utilities (src/file_utils.py)"]
        GeneralUtils["General Utils (src/utils.py)"]
    end

    subgraph "Presentation Layer"
        RichDisplay["Rich Display (src/rich_display.py)"]
    end

    CLI --> BatchRunner
    BatchRunner --> FileProcessor
    BatchRunner --> InputSplitter
    BatchRunner --> DataLoader
    BatchRunner --> RichDisplay
    BatchRunner --> ConfigLoader
    BatchRunner --> LoggerUtil

    FileProcessor --> LLMClient
    FileProcessor --> BatchJobClass
    FileProcessor --> TokenUtils
    FileProcessor --> TokenTracker
    FileProcessor --> DataLoader
    FileProcessor --> PromptUtils
    FileProcessor --> InputSplitter

    LLMClient --> OpenAI_API
    LLMClient --> TokenTracker

    InputSplitter --> DataLoader
    InputSplitter --> TokenUtils

    TokenTracker --> CostEstimator

    ConfigLoader --> Constants
    ConfigLoader --> GeneralUtils
    ConfigLoader --> PromptUtils

    PromptUtils --> Evaluator

    CLI --> LoggerUtil
```

### 1.3 Component Relationships (Claude's Version)

```mermaid
graph TD
    CLI[cli.py] --> BR[batch_runner.py]
    BR --> FP[file_processor.py]
    BR --> IS[input_splitter.py]
    BR --> TT[token_tracker.py]
    BR --> LC[llm_client.py]

    FP --> BJ[batch_job.py]
    FP --> LC
    FP --> IS
    FP --> TT

    BJ --> TT
    LC --> TT
    LC --> CE[cost_estimator.py]

    CL[config_loader.py] --> BR
    CL --> FP
    CL --> LC

    CONST[constants.py] --> BR
    CONST --> FP
    CONST --> LC
    CONST --> TT

    LOG[logger.py] --> BR
    LOG --> FP
    LOG --> LC
    LOG --> TT

    PU[prompt_utils.py] --> BR
    PU --> EVAL[evaluator.py]

    RD[rich_display.py] --> BR
    RD --> FP

    UTILS[utils.py] --> BR
    UTILS --> FP
    UTILS --> CL

    DL[data_loader.py] --> FP
    DL --> IS

    EXCP[exceptions.py] --> FP
    EXCP --> IS
```

## 2. Data Flow & Processing

### 2.1 Data Flow Diagram

```mermaid
flowchart TD
    A[User] --> B{Executes CLI command};
    B --> C[src/cli.py:main];
    C --> D[src/config_loader.py:load_config];
    D --> E[Config Object];
    E --> F[src/logger.py:setup logging];
    E --> G[src/batch_runner.py:run_batch_processing];
    G --> H[Resolve input files];
    H --> I[Setup output directory];
    I --> J[Process files];
    J --> K[Check token limits];
    K --> L[Prepare output path];
    L --> M[Process file wrapper];
    M --> N[LLMClient:run_batch_job];
    N --> O[OpenAI API];
    O --> P[Process results];
    P --> Q[Save output];
    Q --> R[Report batch results];
```

### 2.2 Data Flow with Batch Processing (Claude's Version)

```mermaid
flowchart TD
    InputFiles[Input Files] -->|Load| FP[File Processor]
    FP -->|chunk| BJ[BatchJobs]
    BJ -->|for each| LC[LLMClient]
    LC -->|request/response| EXT[OpenAI API]
    BJ -->|collect| FP
    FP -->|aggregate| OutputFiles[Output Files]
    FP -->|log| TT[TokenTracker]
    FP -->|log| LOG[Logger]
```

```mermaid
flowchart TD
    A[User] --> B{Executes CLI command};
    B --> C[src/cli.py:main];
    C --> D[src/config_loader.py:load_config];
    D --> E[Config Object];
    E --> F[src/logger.py:setup logging];
    E --> G[src/batch_runner.py:run_batch_processing];
    G --> H[Resolve input files];
    H --> I[Setup output directory];
    I --> J[Process files];
    J --> K[Check token limits];
    K --> L[Prepare output path];
    L --> M[Process file wrapper];
    M --> N[LLMClient:run_batch_job];
    N --> O[OpenAI API];
    O --> P[Process results];
    P --> Q[Save output];
    Q --> R[Report batch results];
```

## 3. Component Relationships & State Management

```mermaid
graph TD
    A_CLI["src/cli.py"] --> B_BatchRunner["src/batch_runner.py"]
    A_CLI --> C_ConfigLoader["src/config_loader.py"]
    A_CLI --> D_Constants["src/constants.py"]
    A_CLI --> E_Logger["src/logger.py"]

    B_BatchRunner --> C_ConfigLoader
    B_BatchRunner --> D_Constants
    B_BatchRunner --> F_DataLoader["src/data_loader.py"]
    B_BatchRunner --> G_FileProcessor["src/file_processor.py"]
    B_BatchRunner --> H_InputSplitter["src/input_splitter.py"]
    B_BatchRunner --> I_LLMClient["src/llm_client.py"]
    B_BatchRunner --> J_LogUtils["src/log_utils.py"]
    B_BatchRunner --> K_PromptUtils["src/prompt_utils.py"]
    B_BatchRunner --> L_TokenTracker["src/token_tracker.py"]
    B_BatchRunner --> M_Utils["src/utils.py"]
    B_BatchRunner --> E_Logger
```

### 3.1 BatchJob State Transitions

```mermaid
stateDiagram-v2
    [*] --> Pending
    Pending --> Running : start()
    Running --> Success : all rows processed
    Running --> Failure : error & halt_on_chunk_failure?
    Running --> Pending : retry
    Failure --> [*]
    Success --> [*]
```

## 4. Execution Paths

### 4.1 Concurrent Processing Flow

```mermaid
sequenceDiagram
    participant CLI
    participant Runner
    participant Processor
    participant Executor
    participant LLM
    participant OpenAI

    CLI->>Runner: args
    Runner->>Processor: filepath, config
    Processor->>Executor: submit_job(chunk)
    Executor->>LLM: run_batch_job()
    LLM->>OpenAI: POST /batch
    OpenAI-->>LLM: result_file_id
    LLM-->>Executor: results
    Executor-->>Processor: aggregated DataFrame
    Processor-->>Runner: success/fail stats
```

### 4.2 Execution Path Overview

- **CLI** → **Config Load** → **Mode Dispatch** → **File Discovery** → **File Processing** → **LLM API** → **Result Aggregation** → **Output/Log/Cost**

#### Error/Recovery Paths

- Token limit exceeded → abort or skip (configurable)
- API/network error → per-chunk fail, aggregate with error message
- File IO error → skip file, halt batch (configurable)
- All errors logged with configurable recovery options

## 5. Token Management & Processing

### 5.1 Token Counting and Chunking Flow

```mermaid
flowchart TD
    A[Start] --> B[Load Input File]
    B --> C{Token Count > Limit?}
    C -->|Yes| D[Split File]
    D --> E[Create Chunks]
    E --> F[Process Chunks]
    C -->|No| G[Process Single Chunk]
    F --> H[Aggregate Results]
    G --> H
    H --> I[Save Output]
    I --> J[End]
```

### 5.2 Token Management System

```mermaid
classDiagram
    class TokenTracker {
        +log_token_usage_event()
        +update_token_log()
        +get_token_usage()
        +calculate_costs()
    }

    class CostEstimator {
        +get_model_cost()
        +estimate_batch_cost()
    }

    class LLMClient {
        +run_batch_job()
        +_prepare_batch_requests()
        +_manage_batch_job()
    }

    TokenTracker --> CostEstimator
    LLMClient --> TokenTracker
```

## 6. Error Handling & Recovery

### 6.1 Error Handling Flow

```mermaid
flowchart TD
    A[Start Processing] --> B{API Call}
    B -->|Success| C[Process Response]
    B -->|Error| D{Retry?}
    D -->|Yes| B
    D -->|No| E[Log Error]
    E --> F[Handle Failure]
    C --> G[Next Chunk]
    F --> G
    G --> H{More Chunks?}
    H -->|Yes| B
    H -->|No| I[End Processing]
```

### 6.2 Error Classification

- **Input Validation Errors**: Invalid file formats, missing columns
- **API Errors**: Rate limits, authentication, timeouts
- **Processing Errors**: Token limits, malformed outputs
- **System Errors**: File I/O, memory issues

Each error type has specific handling strategies and recovery mechanisms.

## 7. Configuration Management

### 7.1 Configuration Loading Flow

```mermaid
flowchart TD
    A[Start] --> B[Load Default Config]
    B --> C[Load User Config]
    C --> D[Load Environment Vars]
    D --> E[Merge Configs]
    E --> F[Validate Config]
    F --> G[Return Config]
    F -->|Error| H[Log Error & Exit]
    G --> I[End]
```

### 7.2 Configuration Layers

1. **Default Configuration**: Hardcoded defaults in `constants.py`
2. **User Configuration**: `config.yaml` in project root
3. **Environment Variables**: Override specific settings
4. **Command-Line Arguments**: Highest precedence

### 7.3 Configuration Schema

```yaml
# Example Configuration Structure
batch_processing:
  max_workers: 4
  chunk_size: 1000
  max_retries: 3

llm:
  model: gpt-4
  temperature: 0.7
  max_tokens: 2048

token_management:
  token_limit: 1000000
  cost_tracking: true

logging:
  level: INFO
  file: batch_grader.log
  max_size: 10485760 # 10MB
  backup_count: 5
```

```mermaid
flowchart TD
    A[Start] --> B[Load Default Config]
    B --> C[Load User Config]
    C --> D[Load Environment Vars]
    D --> E[Merge Configs]
    E --> F[Validate Config]
    F --> G[Return Config]
    F -->|Error| H[Log Error & Exit]
    G --> I[End]
```
