# Architecture Diagrams

## 1. Execution Flow ((Sorry, I like graphs))

This diagram provides an overview of the main components and their primary interactions, aligning component names with your Python files and detailing data sources.

```graphviz
digraph UpdatedExecutionFlow {
    rankdir=TB;
    bgcolor="#1E1E1E";
    node [shape=box, style="rounded,filled", fontcolor="white", color="#555555"];
    edge [color="#AAAAAA", fontcolor="#CCCCCC"];

    User [label="User (CLI)", shape=ellipse, fillcolor="#4A235A"];
    Main [label="batch_runner.py\nmain() / argparse", fillcolor="#1A5276"];

    subgraph cluster_Config {
        label="Configuration Loading"; fillcolor="#2C3E50"; fontcolor="#ECF0F1";
        ConfigLoader [label="config_loader.py\nload_config(), ensure_config_files(), is_examples_file_default()", fillcolor="#117A65"];
        FS_Config [label="config.yaml", shape=note, fillcolor="#7D6608"];
        FS_Prompts [label="prompts.yaml", shape=note, fillcolor="#7D6608"];
        FS_Examples_Cfg [label="examples.txt\n(existence/default check)", shape=note, fillcolor="#7D6608"];
    }

    subgraph cluster_CoreProcessing {
        label="Core File Processing Loop (per file)"; fillcolor="#2C3E50"; fontcolor="#ECF0F1";
        ProcessFile [label="batch_runner.py\nprocess_file(filepath)\nOR token counting/splitting logic", fillcolor="#1A5276"];
        DataLoader [label="data_loader.py\nload_data(), save_data()", fillcolor="#117A65"];
        Evaluator [label="evaluator.py\nload_prompt_template()", fillcolor="#117A65"];
        FS_Input [label="input/[file]", shape=note, fillcolor="#7D6608"];
        FS_Examples_Content [label="examples.txt\n(content reading)", shape=note, fillcolor="#7D6608"];
        FS_Output [label="output/[file]", shape=note, fillcolor="#7D6608"];
        TokenCounterFunc [label="batch_runner.py\n(tiktoken based token counting function)", fillcolor="#B3B6B7", fontcolor="black"];
    }

    subgraph cluster_LLM {
        label="LLM Interaction (if not just counting/splitting)"; fillcolor="#2C3E50"; fontcolor="#ECF0F1";
        LLMClient [label="llm_client.py\nLLMClient.run_batch_job()", fillcolor="#117A65"];
        OpenAI_API [label="OpenAI Batch API", shape=ellipse, fillcolor="#4A235A"];
        subgraph cluster_LLM_Internal {
            label="LLMClient Internal Steps"; fillcolor="#5D6D7E"; fontcolor="#ECF0F1";
            Prepare [label="_prepare_batch_requests()"]; Upload [label="_upload_batch_input_file()"];
            Manage [label="_manage_batch_job()"]; ProcessOutputs [label="_process_batch_outputs()"];
        }
    }

    subgraph cluster_Utilities {
        label="Utilities"; fillcolor="#2C3E50"; fontcolor="#ECF0F1";
        CostEstimator [label="cost_estimator.py\nestimate_cost()", fillcolor="#117A65"];
        TokenTracker [label="token_tracker.py\nupdate_token_log(), get_token_usage_for_day()", fillcolor="#117A65"];
        InputSplitter [label="input_splitter.py\nsplit_file_by_token_limit()", fillcolor="#117A65"];
        FS_Pricing [label="docs/pricing.csv", shape=note, fillcolor="#7D6608"];
        FS_TokenLog [label="output/token_usage_log.json", shape=note, fillcolor="#7D6608"];
        FS_SplitOutput [label="input/[file_partN]", shape=note, fillcolor="#7D6608"];
    }

    // Connections
    User -> Main;
    Main -> ConfigLoader [label="Load initial config"];
    ConfigLoader -> FS_Config; ConfigLoader -> FS_Prompts; ConfigLoader -> FS_Examples_Cfg;

    Main -> TokenTracker [label="Display daily usage"];
    TokenTracker -> FS_TokenLog;

    Main -> ProcessFile [label="For each file or specified file"];
    ProcessFile -> DataLoader [label="Load input data"];
    DataLoader -> FS_Input;

    ProcessFile -> ConfigLoader [label="Check examples default status"];
    ProcessFile -> Evaluator [label="Load prompt template"];
    Evaluator -> FS_Prompts;
    ProcessFile -> FS_Examples_Content [label="Read examples content"];
    ProcessFile -> TokenCounterFunc [label="Define/Use token counter"];

    alt "CLI: --count-tokens"
        ProcessFile -> TokenCounterFunc [label="Apply to rows, print stats"];
    else "CLI: --split-tokens"
        ProcessFile -> InputSplitter [label="If tokens > limit"];
        InputSplitter -> TokenCounterFunc [label="Uses"];
        InputSplitter -> FS_Input [label="Reads original"];
        InputSplitter -> FS_SplitOutput [label="Writes parts"];
    else "CLI: Default Run (Batch Job)"
        ProcessFile -> LLMClient;
        LLMClient -> Prepare -> Upload -> Manage -> ProcessOutputs;
        Upload -> OpenAI_API [label="Upload .jsonl"];
        Manage -> OpenAI_API [label="Create/Poll Batch"];
        ProcessOutputs -> OpenAI_API [label="Retrieve Results/Errors"];
        ProcessOutputs -> DataLoader [label="Merge results to DataFrame"];
        ProcessFile -> TokenTracker [label="Update submitted tokens"];
        ProcessFile -> CostEstimator [label="Estimate cost"];
        CostEstimator -> FS_Pricing;
        ProcessFile -> DataLoader [label="Save final output"];
        DataLoader -> FS_Output;
    end
}
```

## 2. Module Interactions & File Access

This diagram focuses on the direct interactions between the core Python modules and the configuration/data files they access.

```graphviz
digraph UpdatedModuleInteractions {
    bgcolor="#1E1E1E";
    node [style="filled,rounded", fontcolor="white", color="#555555", shape=box];
    edge [color="#AAAAAA", fontcolor="#CCCCCC"];

    // External Actor
    User [label="User / CLI", shape=ellipse, fillcolor="#4A235A"];

    // Core Python Modules (src/)
    BR [label="batch_runner.py", fillcolor="#1A5276"];
    CL [label="config_loader.py", fillcolor="#117A65"];
    DL [label="data_loader.py", fillcolor="#117A65"];
    E [label="evaluator.py\n(Prompt Loader)", fillcolor="#117A65"];
    LC [label="llm_client.py", fillcolor="#117A65"];
    CE [label="cost_estimator.py", fillcolor="#117A65"];
    TT [label="token_tracker.py", fillcolor="#117A65"];
    IS [label="input_splitter.py", fillcolor="#117A65"];

    // Configuration & Data Files
    ConfigYAML [label="config/config.yaml", shape=note, fillcolor="#7D6608"];
    PromptsYAML [label="config/prompts.yaml", shape=note, fillcolor="#7D6608"];
    ExamplesTXT [label="examples/examples.txt", shape=note, fillcolor="#7D6608"];
    PricingCSV [label="docs/pricing.csv", shape=note, fillcolor="#7D6608"];
    InputDir [label="input/\n(User Data, Split Outputs)", shape=folder, fillcolor="#7D6608"];
    OutputDir [label="output/\n(Results, Error Logs)", shape=folder, fillcolor="#7D6608"];
    TokenLogJSON [label="output/token_usage_log.json", shape=note, fillcolor="#7D6608"];

    // External Services
    OpenAI_API [label="OpenAI Batch API", shape=ellipse, fillcolor="#4A235A"];

    // Grouping
    subgraph cluster_BatchGraderSystem {
        label="BatchGrader System";
        fontcolor="#CCCCCC"; color="#555555";
        BR; CL; DL; E; LC; CE; TT; IS;
    }
    subgraph cluster_FileSystemData {
        label="File System (Data & Config)";
        fontcolor="#CCCCCC"; color="#555555";
        ConfigYAML; PromptsYAML; ExamplesTXT; PricingCSV; InputDir; OutputDir; TokenLogJSON;
    }

    // Relationships
    User -> BR [label="Initiates via CLI options\n(--count-tokens, --split-tokens, --file, or default run)"];

    BR -> CL [label="Uses"];
    BR -> DL [label="Uses for I/O"];
    BR -> E [label="Uses (load prompt)"];
    BR -> LC [label="Uses (for batch jobs)"];
    BR -> CE [label="Uses (for cost estimate)"];
    BR -> TT [label="Uses (log usage)"];
    BR -> IS [label="Uses (if --split-tokens)"];

    CL -> ConfigYAML [label="Reads/Writes Default"];
    CL -> PromptsYAML [label="Reads/Writes Default"];
    CL -> ExamplesTXT [label="Checks if Default,\nEnsures Exists"];

    E -> PromptsYAML [label="Reads Templates"];
    E -> CL [label="Fallback to Default Prompts"];

    DL -> InputDir [label="Reads From"];
    DL -> OutputDir [label="Writes To (Results, Error Logs)"];

    LC -> OpenAI_API [label="Interacts (Upload, Create, Poll, Retrieve)"];
    LC -> DL [label="Uses (to write temp .jsonl for upload - via tempfile)"]; // Indirectly via tempfile

    CE -> PricingCSV [label="Reads"];
    TT -> TokenLogJSON [label="Reads/Writes"];
    IS -> DL [label="Uses (to read input for splitting)"]; // To read the file to be split
    IS -> InputDir [label="Writes Split Parts To"]; // Writes output parts


    // Data flow for specific operations
    BR -> ExamplesTXT [label="Reads Content"];
    BR -> DL [label="load_data()"];
    BR -> DL [label="save_data()"];
    BR -> IS [label="split_file_by_token_limit()"];
    BR -> TT [label="update_token_log() / get_token_usage_for_day()"];
    BR -> CE [label="estimate_cost()"];
    BR -> LC [label="run_batch_job()"];
}
```

---

## 3. Architecture Diagram (Combined Structure & Execution)

This flowchart offers a more granular view of the execution flow, particularly within the process_file function of batch_runner.py and the internal operations of llm_client.py.

```graphviz
digraph BatchGrader {
  rankdir=TB;
  bgcolor="#1E1E1E";
  node [shape=box, style="rounded,filled", fontcolor="white", color="#555555"];
  edge [color="#AAAAAA", fontcolor="#CCCCCC"];

  // External actor
  User [label="User / System", shape=ellipse, fillcolor="#4A235A"];

  // Main script
  Main [label="batch_runner.py\nmain()", fillcolor="#1A5276"];

  // Config loader
  Config [label="config_loader.py\nConfigLoader.load_config()", fillcolor="#117A65"];
  FS_Config [label="config/config.yaml", shape=note, fillcolor="#7D6608"];

  // Process file
  Proc [label="process_file(filepath)", fillcolor="#1A5276"];
  // Data loader
  DataLoad [label="data_loader.py\nDataLoader.load_data()", fillcolor="#117A65"];
  FS_Input [label="input/* (Data files)", shape=note, fillcolor="#7D6608"];
  FS_Examples [label="examples/examples.txt", shape=note, fillcolor="#7D6608"];

  // Prompt evaluator
  Eval [label="evaluator.py\nEvaluator.load_prompt_template()", fillcolor="#117A65"];
  FS_Prompts [label="config/prompts.yaml", shape=note, fillcolor="#7D6608"];

  // LLM client & batch
  LLM [label="LLMClient", fillcolor="#117A65"];
  subgraph cluster_batch {
    label="llm_client.py – batch ops";
    style=filled; 
    color="#654321"; 
    fillcolor="#5D4037";
    fontcolor="#CCCCCC";
    Prepare [label="_prepare_batch_requests()", fillcolor="#3E2723"];
    Upload  [label="_upload_batch_input_file()", fillcolor="#3E2723"];
    Manage  [label="_manage_batch_job()", fillcolor="#3E2723"];
    Collect [label="_collect_batch_outputs()", fillcolor="#3E2723"];
    OpenAI [label="OpenAI API", shape=ellipse, fillcolor="#4A235A"];

    Prepare -> Upload -> OpenAI;
    Upload -> Manage -> OpenAI;
    Manage -> Collect -> OpenAI;
    Collect -> LLM;
  }

  // Saving & cost
  Save [label="save_data(df, output_path)", fillcolor="#117A65"];
  Cost [label="cost_estimator.py\nCostEstimator.estimate_cost()", fillcolor="#117A65"];
  FS_Pricing [label="docs/pricing.csv", shape=note, fillcolor="#7D6608"];
  Log [label="Console / Log", shape=note, fillcolor="#512E5F"];

  // Error branch
  Error [label="Error Handling\n(save_data → ERROR_*)", fillcolor="#922B21"];
  FS_Error [label="output/ERROR_*", shape=note, fillcolor="#7D6608"];

  // Edges
  User -> Main;
  Main -> Config -> FS_Config;
  Main -> Proc;
  Proc -> DataLoad -> FS_Input -> DataLoad -> Proc;
  Proc -> Config -> Proc [label=" check examples_dir"];
  Proc -> FS_Examples;
  Proc -> Eval -> FS_Prompts -> Eval -> Proc;
  Proc -> LLM -> Prepare;
  Proc -> Save -> FS_Pricing [style=dotted];
  Cost -> FS_Pricing -> Cost -> Proc -> Log;
  Proc -> Save -> FS_Config;
  Proc -> Error -> FS_Error;
}
```

---

## 4. Sequence Diagram (File Processing Flow)

This sequence diagram provides an "insane" level of detail, especially for the interactions within llm_client.py and its communications with the OpenAI API and FileSystem.

```graphviz
digraph ProcessSequence {
    // Dark background and styling
    bgcolor="#1E1E1E";
    node [style="filled,rounded", fontcolor="white", color="#555555"];
    edge [color="#AAAAAA", fontcolor="#CCCCCC"];
    
    // Ranking to maintain sequence diagram flow (top to bottom)
    rankdir=TB;
    
    // Define nodes for participants with different colors
    User [label="User", shape=circle, fillcolor="#4A235A"];
    Runner [label="batch_runner.py", fillcolor="#1A5276"];
    ConfigL [label="config_loader.py", fillcolor="#117A65"];
    DataL [label="data_loader.py", fillcolor="#117A65"];
    Evaluator [label="evaluator.py", fillcolor="#117A65"];
    LLMCli [label="llm_client.py", fillcolor="#117A65"];
    CostEst [label="cost_estimator.py", fillcolor="#117A65"];
    OpenAI [label="OpenAI API", shape=ellipse, fillcolor="#4A235A"];
    FileSystem [label="FileSystem", shape=cylinder, fillcolor="#4A235A"];
    
    // Initial sequence
    Init1 [label="main()", shape=plaintext, fillcolor="transparent"];
    Init2 [label="load_config()", shape=plaintext, fillcolor="transparent"];
    Init3 [label="config", shape=plaintext, fillcolor="transparent"];
    Init4 [label="List files in INPUT_DIR", shape=plaintext, fillcolor="transparent"];
    
    // Main processing subgraph for clarity
    subgraph cluster_main_process {
        label="File Processing Flow";
        fontcolor="#CCCCCC";
        color="#555555";
        style="filled";
        fillcolor="#0E191A";
        
        // Process file activities
        Process1 [label="process_file\n(input_file_path)", fillcolor="#1A5276"];
        LoadData [label="load_data\n(input_file_path)", fillcolor="#117A65"];
        GetExamples [label="is_examples_file_default\n(examples_path)", fillcolor="#117A65"];
        ReadExamples [label="Read examples.txt", fillcolor="#7D6608"];
        LoadPromptTemplate [label="load_prompt_template\n(template_name)", fillcolor="#117A65"];
        FormatPrompt [label="format prompt\nwith examples", fillcolor="#1A5276"];
        
        // LLM client operations
        CreateLLMClient [label="new LLMClient()", fillcolor="#117A65"];
        RunBatchJob [label="run_batch_job\n(df, system_prompt)", fillcolor="#117A65"];
        PrepareRequests [label="_prepare_batch_requests", fillcolor="#3E2723"];
        UploadFile [label="_upload_batch_input_file", fillcolor="#3E2723"];
        ManageJob [label="_manage_batch_job", fillcolor="#3E2723"];
        CollectOutputs [label="_process_batch_outputs", fillcolor="#3E2723"];
        
        // Final operations
        SaveResults [label="save_data\n(df_with_results)", fillcolor="#117A65"];
        EstimateCost [label="estimate_cost\n(model, tokens)", fillcolor="#117A65"];
        LogCost [label="Log estimated cost", fillcolor="#1A5276"];
        ErrorHandling [label="Error Handling", fillcolor="#922B21"];
        SaveError [label="save_data\n(error_df)", fillcolor="#922B21"];
    }
    
    // External resources
    APIFiles [label="files.create\nfiles.content", fillcolor="#4A235A"];
    APIBatches [label="batches.create\nbatches.retrieve", fillcolor="#4A235A"];
    InputFile [label="input/* files", shape=note, fillcolor="#7D6608"];
    ExamplesFile [label="examples/examples.txt", shape=note, fillcolor="#7D6608"];
    PromptsFile [label="config/prompts.yaml", shape=note, fillcolor="#7D6608"];
    OutputFile [label="output/* files", shape=note, fillcolor="#7D6608"];
    PricingFile [label="docs/pricing.csv", shape=note, fillcolor="#7D6608"];
    ErrorFile [label="output/ERROR_*", shape=note, fillcolor="#7D6608"];
    
    // Main flow connections
    User -> Init1 -> Runner;
    Runner -> Init2 -> ConfigL;
    ConfigL -> Init3 -> Runner;
    Runner -> Init4 -> FileSystem;
    
    // Process file flow
    Runner -> Process1 -> LoadData -> DataL;
    DataL -> InputFile [dir=both, label="Read/Parse"];
    DataL -> Runner [label="df (DataFrame)"];
    
    // Empty check branch
    Process1 -> GetExamples -> ConfigL [label="if df not empty"];
    ConfigL -> ExamplesFile [dir=both, label="Check"];
    ConfigL -> Runner [label="is_default"];
    
    // Examples & Prompt flow
    Runner -> ReadExamples -> ExamplesFile [label="if is_default false"];
    Runner -> LoadPromptTemplate -> Evaluator;
    Evaluator -> PromptsFile [dir=both, label="Read"];
    Evaluator -> Runner [label="prompt_template"];
    Runner -> FormatPrompt [label="format template"];
    
    // LLM operations flow
    Runner -> CreateLLMClient -> LLMCli;
    LLMCli -> Runner [label="client instance"];
    Runner -> RunBatchJob -> LLMCli;
    LLMCli -> PrepareRequests [label="prepare"];
    PrepareRequests -> UploadFile [label="upload"];
    UploadFile -> FileSystem [label="Write temp .jsonl"];
    UploadFile -> APIFiles -> OpenAI [label="upload"];
    OpenAI -> LLMCli [label="input_file_id"];
    UploadFile -> ManageJob [label="manage"];
    ManageJob -> APIBatches -> OpenAI [label="create & poll"];
    OpenAI -> LLMCli [label="batch status"];
    ManageJob -> CollectOutputs [label="process"];
    CollectOutputs -> OpenAI [label="retrieve results"];
    OpenAI -> LLMCli [label="output/error files"];
    CollectOutputs -> LLMCli [label="results"];
    LLMCli -> Runner [label="df_with_results"];
    
    // Save results & Cost estimation
    Runner -> SaveResults -> DataL;
    DataL -> OutputFile [label="Write"];
    DataL -> Runner [label="save status"];
    Runner -> EstimateCost -> CostEst [label="if tiktoken available"];
    CostEst -> PricingFile [dir=both, label="Read"];
    CostEst -> Runner [label="cost estimate"];
    Runner -> LogCost [label="log"];
    
    // Error handling path
    Process1 -> ErrorHandling [style=dashed, label="On Exception"];
    ErrorHandling -> SaveError -> DataL;
    DataL -> ErrorFile [label="Write"];
    DataL -> Runner [label="error status"];
    Runner -> User [label="Processing Finished"];
    
    // Visual organization - group related nodes
    {rank=same; User Init1}
    {rank=same; Runner Init2}
    {rank=same; ConfigL Init3}
    {rank=same; FileSystem Init4}
    {rank=same; Process1 LoadData}
    {rank=same; DataL InputFile}
    {rank=same; GetExamples ConfigL}
    {rank=same; LoadPromptTemplate Evaluator PromptsFile}
    {rank=same; RunBatchJob LLMCli}
    {rank=same; APIFiles OpenAI APIBatches}
    {rank=same; SaveResults DataL OutputFile}
    {rank=same; EstimateCost CostEst PricingFile}
    {rank=same; ErrorHandling SaveError ErrorFile}
}
