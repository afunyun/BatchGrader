import os
import uuid
import datetime
import argparse

from pathlib import Path
from data_loader import load_data, save_data
from evaluator import load_prompt_template
from config_loader import load_config, CONFIG_DIR, is_examples_file_default
from llm_client import LLMClient
from cost_estimator import CostEstimator
from token_tracker import update_token_log, get_token_usage_for_day
from input_splitter import split_file_by_token_limit

config = load_config()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = str(PROJECT_ROOT / config['input_dir'])
OUTPUT_DIR = str(PROJECT_ROOT / config['output_dir'])
RESPONSE_FIELD = config['response_field']
TOKEN_LIMIT = config.get('token_limit', 2_000_000)
split_token_limit = config.get('split_token_limit', 500_000)

model_name = config['openai_model_name']

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_file(filepath):
    """
    Processes a single input file using the OpenAI Batch API workflow via LLMClient.
    Loads data, prepares batch requests, manages the batch job, processes results, and saves output.
    Handles errors and logs appropriately.
    Enforces configured token limit per batch, halts and warns if exceeded, and logs daily submitted tokens per API key (censored) in output/token_usage_log.json.
    Returns:
        True if successful, False if any error or batch job failure.
    Args:
        filepath (str): Path to the input file to process.
    """
    filename = os.path.basename(filepath)
    base_output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(base_output_path):
        file_root, file_ext = os.path.splitext(filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
            raise RuntimeError("#~# you need tiktoken or half of the functionality explodes my guy, rerun uv pip install -r requirements.txt #~#")

        def count_input_tokens_per_row(row, system_prompt_content, response_field, enc):
            sys_tokens = len(enc.encode(system_prompt_content))
            user_prompt = f"Please evaluate the following text: {str(row[response_field])}"
            user_tokens = len(enc.encode(user_prompt))
            return sys_tokens + user_tokens

        def count_completion_tokens(row, enc):
            completion = row.get('llm_score', '')
            return len(enc.encode(str(completion)))

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

        if total_tokens > TOKEN_LIMIT:
            print(f"[ERROR] Total submitted tokens ({total_tokens}) exceeds the allowed cap of {TOKEN_LIMIT} for a single batch.")
            print("Please reduce your batch size or check your usage at https://platform.openai.com/usage.")
            print("Batch submission halted. No API calls were made.")
            try:
                llm_client = LLMClient()
                update_token_log(llm_client.api_key, 0)
            except Exception as e:
                print(f"[WARN] Could not log token usage: {e}")
            return False
        try:
            llm_client = LLMClient()
            update_token_log(llm_client.api_key, int(total_tokens))
        except Exception as e:
            print(f"[WARN] Could not log token usage: {e}")

        if df.empty:
            print(f"No data loaded from {filepath}. Skipping.")
            return False

        MAX_BATCH_SIZE = 50000
        if len(df) > MAX_BATCH_SIZE:
            print(f"[WARN] Input file contains {len(df)} rows. Only the first {MAX_BATCH_SIZE} will be sent to the API (limit is 50,000 per batch). The rest will be ignored for this run. Simultaneous requests to the API are not supported yet but I am working on it.")
            df = df.iloc[:MAX_BATCH_SIZE].copy() ##TODO - submit the split batches and monitor all simultaneously.

        llm_client = LLMClient()
        try:
            df_with_results = llm_client.run_batch_job(
                df, system_prompt_content, response_field_name=RESPONSE_FIELD, base_filename_for_tagging=filename
            )
        except Exception as batch_exc:
            print(f"[ERROR] Batch job failed for {filepath}: {batch_exc}")
            return False
        save_data(df_with_results.drop(columns=['custom_id'], errors='ignore'), output_path)
        print(f"Processed {filepath}. Results saved to {output_path}")
        print(f"Total rows successfully processed: {len(df_with_results)}")
        error_rows = df_with_results['llm_score'].str.contains('Error', case=False)
        if error_rows.any():
            print(f"Total rows with errors: {error_rows.sum()}")
            if error_rows.sum() == len(df_with_results):
                print(f"[BATCH FAILURE] All rows failed for {filepath}. Halting further processing.")
                return False
        
        try:
            input_col_name = 'input_tokens'
            output_col_name = 'output_tokens'
            if enc is None and (input_col_name not in df_with_results.columns or output_col_name not in df_with_results.columns):
                raise RuntimeError("#~# you need tiktoken or half of the functionality explodes my guy, rerun uv pip install -r requirements.txt #~#")
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
        log_basename = os.path.splitext(filename)[0] + ".log"
        log_base_path = os.path.join(OUTPUT_DIR, log_basename)
        if os.path.exists(log_base_path):
            file_root, file_ext = os.path.splitext(log_basename)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_log_filename = f"{file_root}_{timestamp}{file_ext}"
            log_path = os.path.join(OUTPUT_DIR, new_log_filename)
            print(f"Log file already exists. Using new filename: {new_log_filename}")
        else:
            log_path = log_base_path
        try:
            with open(log_path, 'w') as f_log:
                f_log.write(f"Failed to process {filepath}.\nError: {e}\n")
            print(f"Logged error to {log_path}")
        except Exception as save_err:
            print(f"Double fail: exception in processing and then in saving error log for {filepath}: {save_err} ... aborting.")
        return False

    return True

def get_request_mode(args):
    """
    Indicates in the CLI whether or not requests are being submitted or we're in a safe count/split mode.
    """
    if getattr(args, 'count_tokens', False) or getattr(args, 'split_tokens', False):
        return "Split/Count (NO REQUESTS MADE)"
    return "API Request/Batch"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BatchGrader CLI: batch LLM evaluation, token counting, and input splitting.")
    parser.add_argument('--count-tokens', action='store_true', help='Count tokens in input file(s) and print stats.')
    parser.add_argument('--split-tokens', action='store_true', help='Split input file(s) into parts not exceeding the configured token limit.')
    parser.add_argument('--file', type=str, default=None, help='Only process the specified file in the input directory.')
    args = parser.parse_args()

    llm_client = LLMClient()
    if not llm_client.api_key:
        print("Error: OPENAI_API_KEY not set in config/config.yaml.")
        exit(1)
    else:
        tokens_today = get_token_usage_for_day(llm_client.api_key)
        print("\n================= API USAGE =================")
        print(f"TOTAL TOKENS SUBMITTED TODAY: {tokens_today:,}")
        print(f"TOKEN LIMIT: {TOKEN_LIMIT}")
        print(f"TOKENS REMAINING: {TOKEN_LIMIT - tokens_today:,}")
        print(f"SPLIT TOKEN LIMIT: {split_token_limit}")
        print(f"System is running in {get_request_mode(args)} mode.")
        print("=============================================\n")

    print(f"Valid INPUT_DIR: {INPUT_DIR}")
    print(f"Valid OUTPUT_DIR: {OUTPUT_DIR}")

    from pathlib import Path
    def resolve_and_load_input_file(file_arg):
        """
        Resolves the file path for CLI input and loads the data.
        - Absolute path: used as-is
        - Relative with directory: resolved from project root
        - Bare filename: resolved from input dir
        Returns (resolved_path, DataFrame)
        """
        file_arg_path = Path(file_arg)
        if file_arg_path.is_absolute():
            resolved_path = str(file_arg_path)
        elif file_arg_path.parent != Path('.'):
            resolved_path = str((PROJECT_ROOT / file_arg_path).resolve())
        else:
            resolved_path = os.path.join(INPUT_DIR, file_arg)
        if not os.path.exists(resolved_path):
            print(f"File {file_arg} not found at {resolved_path}.")
            exit(1)
        df = load_data(resolved_path)
        return resolved_path, df

    files_found = []
    if args.file:
        resolved_path, df = resolve_and_load_input_file(args.file)
        files_found = [(resolved_path, df)]
    else:
        files_found = []
        for file_to_process in os.listdir(INPUT_DIR):
            if file_to_process.endswith((".csv", ".json", ".jsonl")):
                resolved_path, df = resolve_and_load_input_file(file_to_process)
                files_found.append((resolved_path, df))
    if not files_found:
        print(f"Nothing found in {INPUT_DIR} (looked for .csv, .json, .jsonl, if your data isn't in one of these formats I'm both worried and impressed, please reformat.)")
        exit(0)

    def get_token_counter(system_prompt_content, response_field, enc):
        def count_submitted_tokens(row):
            sys_tokens = len(enc.encode(system_prompt_content))
            user_prompt = f"Please evaluate the following text: {str(row[response_field])}"
            user_tokens = len(enc.encode(user_prompt))
            return sys_tokens + user_tokens
        return count_submitted_tokens

    # This prevents submitting further batches if one fails (2025-05-11). Halts immediately. RIP my poor API credits.
    for resolved_path, df in files_found:
        try:
            print(f"\nProcessing file: {resolved_path}")
            
            examples_dir = config.get('examples_dir')
            if not examples_dir:
                raise ValueError("'examples_dir' not found in config.yaml.")
            project_root = CONFIG_DIR.parent
            abs_examples_path = (project_root / examples_dir).resolve()
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

            if args.count_tokens or args.split_tokens:
                if enc is None:
                    raise RuntimeError("#~# you need tiktoken or half of the functionality explodes my guy, rerun uv pip install -r requirements.txt #~#")
                token_counter = get_token_counter(system_prompt_content, RESPONSE_FIELD, enc)
                token_counts = df.apply(token_counter, axis=1)
                total_tokens = token_counts.sum()
                avg_tokens = token_counts.mean()
                max_tokens = token_counts.max()
                print(f"[TOKEN COUNT] Total: {total_tokens}, Avg: {avg_tokens:.1f}, Max: {max_tokens}")
                if args.split_tokens:
                    display_name = os.path.basename(resolved_path)
                    if total_tokens <= TOKEN_LIMIT:
                        print(f"File {display_name} does not exceed the token limit. No split needed.")
                    else:
                        print(f"Splitting {display_name} into chunks not exceeding {split_token_limit} tokens...")
                        output_files, token_counts = split_file_by_token_limit(resolved_path, split_token_limit, token_counter, RESPONSE_FIELD, output_dir=INPUT_DIR)
                        print(f"Split complete. Output files: {output_files}")
                        for out_file, tok_count in zip(output_files, token_counts):
                            print(f"Output file: {out_file} | Tokens: {tok_count}")
                continue

            ok = process_file(resolved_path)
            if not ok:
                print(f"[BATCH HALTED] Halting further batch processing due to failure in {resolved_path}.")
                break
        except Exception as e:
            print(f"[BATCH HALTED] Error processing {resolved_path}: {e}")
            print("Halting further batch processing due to failure.")
            break
    print("Batch finished processing.")
