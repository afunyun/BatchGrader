import os
import uuid
import datetime

from pathlib import Path
from data_loader import load_data, save_data
from evaluator import load_prompt_template
from config_loader import load_config, CONFIG_DIR, is_examples_file_default
from llm_client import LLMClient
from cost_estimator import CostEstimator

config = load_config()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = str(PROJECT_ROOT / config['input_dir'])
OUTPUT_DIR = str(PROJECT_ROOT / config['output_dir'])
RESPONSE_FIELD = config['response_field']

model_name = config['openai_model_name']

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_file(filepath):
    """
    Processes a single input file using the OpenAI Batch API workflow via LLMClient.
    Loads data, prepares batch requests, manages the batch job, processes results, and saves output.
    Handles errors and logs appropriately.

    Args:
        filepath (str): Path to the input file to process.
    """
    filename = os.path.basename(filepath)
    base_output_path = os.path.join(OUTPUT_DIR, filename)
    
    # Check if output file already exists and generate unique filename if needed
    if os.path.exists(base_output_path):
        # Get file extension (if any)
        file_root, file_ext = os.path.splitext(filename)
        # Generate a timestamp string
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a new filename with timestamp
        new_filename = f"{file_root}_{timestamp}{file_ext}"
        output_path = os.path.join(OUTPUT_DIR, new_filename)
        print(f"Output file already exists. Using new filename: {new_filename}")
    else:
        output_path = base_output_path

    print(f"Processing file: {filepath}")
    df = None
    try:
        df = load_data(filepath)
        print(f"Loaded {len(df)} rows from {filepath}")

        examples_dir = config.get('examples_dir')
        if not examples_dir:
            raise ValueError("'examples_dir' not found in config.yaml.")
        project_root = CONFIG_DIR.parent
        abs_examples_path = (project_root / examples_dir).resolve()
        if not abs_examples_path.exists():
            raise FileNotFoundError(f"Examples file not found: {abs_examples_path}")

        if is_examples_file_default(abs_examples_path):
            system_prompt_content = load_prompt_template("batch_evaluation_prompt_generic")
        else:
            with open(abs_examples_path, 'r', encoding='utf-8') as ex_f:
                example_lines = [line.strip() for line in ex_f if line.strip()]
            if not example_lines:
                raise ValueError(f"No examples found in {abs_examples_path}.")
            formatted_examples = '\n'.join(f"- {ex}" for ex in example_lines)
            system_prompt_template = load_prompt_template("batch_evaluation_prompt")
            if '{dynamic_examples}' not in system_prompt_template:
                raise ValueError("'{dynamic_examples}' placeholder missing in prompt template.")
            system_prompt_content = system_prompt_template.format(dynamic_examples=formatted_examples)

        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(config.get('openai_model_name', 'gpt-4o-mini-2024-07-18'))
        except Exception:
            print("[WARN] tiktoken not available, cannot count tokens accurately.")
            enc = None

        def count_input_tokens_per_row(row, system_prompt_content, response_field, enc):
            if enc is None:
                return 0
            sys_tokens = len(enc.encode(system_prompt_content))
            user_prompt = f"Please evaluate the following text: {str(row[response_field])}"
            user_tokens = len(enc.encode(user_prompt))
            return sys_tokens + user_tokens

        def count_completion_tokens(row, enc):
            if enc is None:
                return 0
            completion = row.get('llm_score', '')
            return len(enc.encode(str(completion)))

        if enc is not None:
            def count_submitted_tokens(row):
                sys_tokens = len(enc.encode(system_prompt_content))
                user_prompt = f"Please evaluate the following text: {str(row[RESPONSE_FIELD])}"
                user_tokens = len(enc.encode(user_prompt))
                return sys_tokens + user_tokens
            token_counts = df.apply(count_submitted_tokens, axis=1)
            total_tokens = token_counts.sum()
            avg_tokens = token_counts.mean()
            max_tokens = token_counts.max()
            print(f"[SUBMITTED TOKENS] Total: {total_tokens}, Avg: {avg_tokens:.1f}, Max: {max_tokens}")
        else:
            print("[WARN] Token counting skipped (tiktoken unavailable or model unknown).")

        if df.empty:
            print(f"No data loaded from {filepath}. Skipping.")
            return

        MAX_BATCH_SIZE = 50000
        if len(df) > MAX_BATCH_SIZE:
            print(f"[WARN] Input file contains {len(df)} rows. Only the first {MAX_BATCH_SIZE} will be sent to the API (limit is 50,000 per batch). The rest will be ignored for this run. Simultaneous requests to the API are not supported yet but I am working on it.")
            df = df.iloc[:MAX_BATCH_SIZE].copy() ##TODO - split into multiple batches and monitor all.

        llm_client = LLMClient()
        df_with_results = llm_client.run_batch_job(
            df, system_prompt_content, response_field_name=RESPONSE_FIELD, base_filename_for_tagging=filename
        )
        save_data(df_with_results.drop(columns=['custom_id'], errors='ignore'), output_path)
        print(f"Processed {filepath}. Results saved to {output_path}")
        print(f"Total rows successfully processed: {len(df_with_results)}")
        error_rows = df_with_results['llm_score'].str.contains('Error', case=False)
        if error_rows.any():
            print(f"Total rows with errors: {error_rows.sum()}")
        
        try:
            input_col_name = 'input_tokens'
            output_col_name = 'output_tokens'
            if enc is None and (input_col_name not in df_with_results.columns or output_col_name not in df_with_results.columns):
                raise RuntimeError("tiktoken is required for cost estimation but is not installed.")
            if enc is not None and (input_col_name not in df_with_results.columns or output_col_name not in df_with_results.columns):
                df_with_results[input_col_name] = df_with_results.apply(lambda row: count_input_tokens_per_row(row, system_prompt_content, RESPONSE_FIELD, enc), axis=1)
                df_with_results[output_col_name] = df_with_results.apply(lambda row: count_completion_tokens(row, enc), axis=1)
            n_input_tokens = int(df_with_results[input_col_name].sum()) if input_col_name in df_with_results.columns else 0
            n_output_tokens = int(df_with_results[output_col_name].sum()) if output_col_name in df_with_results.columns else 0
            model_name = config.get('openai_model_name', 'gpt-4o-2024-08-06')
            try:
                cost = CostEstimator.estimate_cost(model_name, n_input_tokens, n_output_tokens)
                print(f"Estimated LLM cost: ${cost:.4f} (input: {n_input_tokens} tokens, output: {n_output_tokens} tokens, model: {model_name})")
            except Exception as ce:
                print(f"Could not estimate cost: {ce}")
        except Exception as cost_exception:
            print(f"[Cost Estimation Error] {cost_exception}")


    except Exception as e:
        print(f"An error occurred while processing {filepath}: {e}")
        error_basename = f"ERROR_{filename}"
        error_base_path = os.path.join(OUTPUT_DIR, error_basename)
        
        # Check if error file already exists and generate unique filename if needed
        if os.path.exists(error_base_path):
            file_root, file_ext = os.path.splitext(error_basename)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_error_filename = f"{file_root}_{timestamp}{file_ext}"
            error_df_path = os.path.join(OUTPUT_DIR, new_error_filename)
            print(f"Error output file already exists. Using new filename: {new_error_filename}")
        else:
            error_df_path = error_base_path
        try:
            if df is not None and not df.empty:
                if 'custom_id' not in df.columns:
                    df['custom_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
                df['llm_score'] = f"Error during processing: {e}"
                save_data(df.drop(columns=['custom_id'], errors='ignore'), error_df_path)
                print(f"Saved error information to {error_df_path}")
            else:
                with open(error_df_path, 'w') as f_err:
                    f_err.write(f"Failed to process {filepath}.\nError: {e}\n")
                print(f"Logged error to {error_df_path} as DataFrame was not available/processed.")

        except Exception as save_err:
            print(f"Rare double fail, exception in processing and then exception to save error state for {filepath}: {save_err}... it's over.")

if __name__ == "__main__":
    print("Starting batch_runner.py...")
    print(f"Valid INPUT_DIR: {INPUT_DIR}")
    print(f"Valid OUTPUT_DIR: {OUTPUT_DIR}")
    llm_client = LLMClient()
    if not llm_client.api_key:
        print("Error: OPENAI_API_KEY not set in config/config.yaml.")
    else:
        files_found = [file_to_process for file_to_process in os.listdir(INPUT_DIR)
                    if file_to_process.endswith((".csv", ".json", ".jsonl"))]
        if not files_found:
            print(f"Nothing found in {INPUT_DIR} (looked for .csv, .json, .jsonl)")
        for file_to_process in files_found:
            full_filepath = os.path.join(INPUT_DIR, file_to_process)
            process_file(full_filepath)
        print("batch_runner.py finished processing.")
