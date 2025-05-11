# BatchGrader Application Flow

## *Version: v0.2.6-pre*

---

## Table of Contents

1. [Overview](#overview)
2. [Component Architecture](#component-architecture)
3. [Main Flow Sequence](#main-flow-sequence)
4. [Data Flow](#data-flow)
5. [Component Descriptions](#component-descriptions)
6. [Configuration Files](#configuration-files)
7. [Workflow Modes](#workflow-modes)

---

## Overview

BatchGrader is a modular Python application for batch evaluation of text data using the OpenAI Batch API. It features YAML-based configuration, prompt management, flexible input/output support, and robust cost estimation.

---

## Component Architecture

The following diagram shows the main components of BatchGrader and their relationships:

```mermaid
%%{init: { 'theme': 'dark', 'themeVariables': {
      'background': '#0d1117',
      'primaryColor': '#21262d',
      'primaryTextColor': '#c9d1d9',
      'primaryBorderColor': '#8b949e',
      'lineColor': '#8b949e',
      'textColor': '#c9d1d9'
}}}%%
classDiagram
    class batchrunner {
        +processfile(filepath)
        +getrequestmode(args)
        +printtokencoststats()
        +printtokencostsummary(summary)
        +resolveandloadinputfile(filearg)
        +gettokencounter(systempromptcontent, responsefield, enc)
        +main()
    }
    class configloader {
        +DEFAULTCONFIG
        +DEFAULTPROMPTS
        +ensureconfigfiles()
        +isexamplesfiledefault(examplespath)
        +loadconfig()
    }
    class dataloader {
        +loaddata(filepath)
        +savedata(df, filepath)
    }
    class evaluator {
        +loadprompttemplate(name)
    }
    class inputsplitter {
        +splitfilebytokenlimit(inputpath, tokenlimit, counttokensfn, responsefield, rowlimit, outputdir, fileprefix)
    }
    class llmclient {
        -apikey
        -model
        -endpoint
        -maxtokens
        -pollinterval
        -client
        +init(model, apikey, endpoint)
        -preparebatchrequests(df, systempromptcontent, responsefieldname)
        -uploadbatchinputfile(requestsdata, basefilenamefortagging)
        -managebatchjob(inputfileid, sourcefilename)
        -processbatchoutputs(batchjobobj, dfwithcustomids)
        +runbatchjob(df, systempromptcontent, responsefieldname, basefilenamefortagging)
    }
    class tokentracker {
        +updatetokenlog(apikey, tokenssubmitted, datestr)
        +logtokenusageevent(apikey, model, inputtokens, outputtokens, timestamp, requestid)
        +loadtokenusageevents()
        +gettokenusagesummary(startdate, enddate, groupby)
        +gettotalcost(startdate, enddate)
        +gettokenusageforday(apikey, datestr)
    }
    class CostEstimator {
        -pricing
        -csvpath
        -loadpricing()
        +estimatecost(model, ninputtokens, noutputtokens)
    }
    batchrunner --> configloader : uses
    batchrunner --> dataloader : uses
    batchrunner --> evaluator : uses
    batchrunner --> inputsplitter : uses
    batchrunner --> llmclient : uses
    batchrunner --> tokentracker : uses
    batchrunner --> CostEstimator : uses
    llmclient --> configloader : uses
    tokentracker --> CostEstimator : uses pricing data
    inputsplitter --> configloader : uses
    evaluator --> configloader : uses
```

---

## Main Flow Sequence

The sequence diagram below illustrates the main workflow from user invocation to result reporting. Conditional branches are shown for different operation modes.

```mermaid
%%{init: { 'theme': 'dark' } }%%
sequenceDiagram
    participant User
    participant batchrunner
    participant configloader
    participant dataloader
    participant evaluator
    participant llmclient
    participant inputsplitter
    participant tokentracker
    participant CostEstimator

    User->>batchrunner: Run with arguments
    batchrunner->>configloader: loadconfig()
    configloader-->>batchrunner: config

    alt --count-tokens flag
        batchrunner->>dataloader: loaddata(filepath)
        dataloader-->>batchrunner: dataframe
        batchrunner->>evaluator: loadprompttemplate()
        evaluator-->>batchrunner: prompttemplate
        batchrunner->>batchrunner: Count tokens
        batchrunner-->>User: Display token counts
    else --split-tokens flag
        batchrunner->>dataloader: loaddata(filepath)
        dataloader-->>batchrunner: dataframe
        batchrunner->>evaluator: loadprompttemplate()
        evaluator-->>batchrunner: prompttemplate
        batchrunner->>batchrunner: Count tokens
        batchrunner->>inputsplitter: splitfilebytokenlimit()
        inputsplitter-->>batchrunner: outputfiles, tokencounts
        batchrunner-->>User: Display split results
    else normal batch processing
        batchrunner->>batchrunner: processfile(filepath)
        activate batchrunner
        batchrunner->>dataloader: loaddata(filepath)
        dataloader-->>batchrunner: dataframe
        batchrunner->>evaluator: loadprompttemplate()
        evaluator-->>batchrunner: prompttemplate
        batchrunner->>batchrunner: Count tokens
        batchrunner->>tokentracker: updatetokenlog()
        batchrunner->>llmclient: runbatchjob()
        activate llmclient
        llmclient->>llmclient: preparebatchrequests()
        llmclient->>llmclient: uploadbatchinputfile()
        llmclient->>llmclient: managebatchjob()
        llmclient->>llmclient: processbatchoutputs()
        llmclient-->>batchrunner: dataframe with results
        deactivate llmclient
        batchrunner->>dataloader: savedata()
        batchrunner->>CostEstimator: estimatecost()
        CostEstimator-->>batchrunner: cost
        deactivate batchrunner
        batchrunner->>tokentracker: printtokencoststats()
        tokentracker->>tokentracker: gettokenusagesummary()
        tokentracker-->>batchrunner: summary
        batchrunner-->>User: Display results and stats
    end
```

---

## Data Flow

The data flow diagram below illustrates the flow of data through the application.

### Data Flow Diagram

```mermaid
%%{init: { 'theme': 'dark' } }%%
flowchart TD
    A[Input Files] --> B[batchrunner.py]
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

---

## Component Descriptions

* **batchrunner.py**: The main entry point of the application. Handles CLI arguments, processes input files, and orchestrates the workflow.
* **configloader.py**: Loads configuration from config.yaml and provides default values.
* **dataloader.py**: Handles loading and saving data in different formats (CSV, JSON, JSONL).
* **evaluator.py**: Loads prompt templates from prompts.yaml.
* **inputsplitter.py**: Splits input files into parts that do not exceed a specified token or row limit.
* **llmclient.py**: Interacts with the OpenAI Batch API to submit batch jobs and process results.
* **tokentracker.py**: Tracks and aggregates OpenAI API token usage for both API limit enforcement and historical/cost tracking.
* **costestimator.py**: Estimates API costs based on the pricing data in docs/pricing.csv.

---

## Configuration Files

* **config.yaml**: Contains configuration parameters like input/output directories, model name, token limits, etc.
* **prompts.yaml**: Contains prompt templates for evaluation.
* **examples.txt**: Contains examples of the target style for contextual evaluation.
* **pricing.csv**: Contains pricing data for different OpenAI models.

---

## Workflow Modes

* **Count Tokens**: Counts tokens in input files and displays statistics.
* **Split Tokens**: Splits input files into parts that do not exceed the configured token limit.
* **Batch Processing**: Processes input files using the OpenAI Batch API, saves results, and displays statistics.
