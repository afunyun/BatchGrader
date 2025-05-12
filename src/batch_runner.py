import os
import sys
import datetime
import logging
import tempfile
from pathlib import Path

from log_utils import prune_logs_if_needed
from logger import logger, BatchGraderLogger
from data_loader import load_data, save_data
from evaluator import load_prompt_template
from src.config_loader import load_config, CONFIG_DIR, is_examples_file_default
from src.llm_client import LLMClient
from src.cost_estimator import CostEstimator
from src.token_tracker import update_token_log, get_token_usage_for_day, get_token_usage_summary, log_token_usage_event
from src.input_splitter import split_file_by_token_limit
from src.batch_job import BatchJob
from rich_display import RichJobTable

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'logs')
ARCHIVE_DIR = os.path.join(LOG_DIR, 'archive')
prune_logs_if_needed(LOG_DIR, ARCHIVE_DIR)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _generate_chunk_job_objects(
    original_filepath,
    config,
    system_prompt_content,
    response_field,
    llm_model_name,
    api_key_prefix,
    tiktoken_encoding_func
):
    """
    Splits the original input file into chunks according to config (forced chunking or token-based),
    and returns a list of BatchJob objects for each chunk.

    Args:
        original_filepath (str): Path to the original input file.
        config (dict): Loaded configuration.
        system_prompt_content (str): The prompt to use for all chunks.
        response_field (str): The column to evaluate.
        llm_model_name (str): Model name for LLM API.
        api_key_prefix (str): API key prefix for logging.
        tiktoken_encoding_func: tiktoken.Encoding object for token counting.
    Returns:
        List[BatchJob]: One per chunk, or empty if input is empty.
    """
    import pandas as pd
    import os

    jobs = []
    force_chunk_count = config.get('force_chunk_count', 0)
    input_dir = os.path.dirname(original_filepath)
    chunked_dir = os.path.join(input_dir, '_chunked')
    os.makedirs(chunked_dir, exist_ok=True)
    base_name, ext = os.path.splitext(os.path.basename(original_filepath))
    df = pd.read_csv(original_filepath) if ext.lower() == '.csv' else pd.read_json(original_filepath, lines=ext.lower() == '.jsonl')

    if df.empty:
        return []

    if force_chunk_count and force_chunk_count > 1:
        chunk_size = len(df) // force_chunk_count
        for i in range(force_chunk_count):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < force_chunk_count - 1 else len(df)
            chunk_df = df.iloc[start_idx:end_idx].copy()
            if chunk_df.empty:
                continue
            with tempfile.NamedTemporaryFile(delete=False, dir=chunked_dir, suffix=ext, prefix=f"{base_name}_forcedchunk_{i+1}_of_{force_chunk_count}_") as tmp_f:
                if ext.lower() == '.csv':
                    chunk_df.to_csv(tmp_f.name, index=False)
                else:
                    chunk_df.to_json(tmp_f.name, orient='records', lines=(ext.lower() == '.jsonl'))
                chunk_path = tmp_f.name
            chunk_id_str = f"forced_{i+1}_of_{force_chunk_count}"
            jobs.append(BatchJob(
                chunk_data_identifier=chunk_path,
                chunk_df=chunk_df,
                system_prompt=system_prompt_content,
                response_field=response_field,
                original_source_file=os.path.basename(original_filepath),
                chunk_id_str=chunk_id_str,
                llm_model=llm_model_name,
                api_key_prefix=api_key_prefix
            ))
    else:
        def count_tokens_fn(row):
            sys_tokens = len(tiktoken_encoding_func.encode(system_prompt_content))
            user_prompt = f"Please evaluate the following text: {str(row[response_field])}"
            user_tokens = len(tiktoken_encoding_func.encode(user_prompt))
            return sys_tokens + user_tokens
        split_token_limit = config.get('split_token_limit', 500_000)
        split_row_limit = config.get('split_row_limit', None)
        output_dir = input_dir
        file_prefix = f"{base_name}_split"
        chunk_files, _ = split_file_by_token_limit(
            original_filepath,
            token_limit=split_token_limit,
            count_tokens_fn=count_tokens_fn,
            response_field=response_field,
            row_limit=split_row_limit,
            output_dir=output_dir,
            file_prefix=file_prefix
        )
        for idx, chunk_path in enumerate(chunk_files):
            chunk_ext = os.path.splitext(chunk_path)[1].lower()
            chunk_df = pd.read_csv(chunk_path) if chunk_ext == '.csv' else pd.read_json(chunk_path, lines=chunk_ext == '.jsonl')
            if chunk_df.empty:
                continue
            chunk_id_str = f"split_{idx+1}_of_{len(chunk_files)}"
            jobs.append(BatchJob(
                chunk_data_identifier=chunk_path,
                chunk_df=chunk_df,
                system_prompt=system_prompt_content,
                response_field=response_field,
                original_source_file=os.path.basename(original_filepath),
                chunk_id_str=chunk_id_str,
                llm_model=llm_model_name,
                api_key_prefix=api_key_prefix
            ))
    return jobs


def _execute_single_batch_job_task(batch_job, llm_client, response_field_name):
    """
    Worker function to process a single BatchJob chunk.
    Updates batch_job status and results in place.
    """
    try:
        batch_job.status = "running"
        df_with_results = llm_client.run_batch_job(
            batch_job.chunk_df,
            batch_job.system_prompt,
            response_field_name=response_field_name,
            base_filename_for_tagging=batch_job.chunk_id_str
        )
        batch_job.results_df = df_with_results
        batch_job.status = "completed"
    except Exception as exc:
        batch_job.status = "failed"
        batch_job.error_message = str(exc)
    return batch_job


def process_file_concurrently(filepath, config, system_prompt_content, response_field, llm_model_name, api_key_prefix, tiktoken_encoding_func):
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from llm_client import LLMClient
    from rich.live import Live

    def _submit_jobs(jobs, response_field):
        llm_client = LLMClient()
        with ThreadPoolExecutor(max_workers=config.get('max_simultaneous_batches', 2)) as executor:
            return {executor.submit(_execute_single_batch_job_task, job, llm_client, response_field): job for job in jobs}

    def _handle_job_result(job, result_job, completed_jobs):
        logger.info(f"Chunk {job.chunk_id_str} started")
        if result_job.status == "running":
            logger.info(f"Chunk {job.chunk_id_str} running")
        if result_job.status == "completed":
            logger.success(f"Chunk {job.chunk_id_str} completed")
        if result_job.status == "failed":
            logger.error(f"Chunk {job.chunk_id_str} failed: {result_job.error_message}")
        completed_jobs.append(result_job)

    def _aggregate_and_cleanup(completed_jobs, filepath):
        result_dfs = [job.results_df for job in completed_jobs if job.status == "completed" and job.results_df is not None]
        if result_dfs:
            combined = pd.concat(result_dfs, ignore_index=True)
            logger.success(f"Results aggregated. {len(combined)} rows processed.")
            from src.file_utils import prune_chunked_dir
            chunked_dir = os.path.join(os.path.dirname(filepath), '_chunked')
            prune_chunked_dir(chunked_dir)
            return combined
        logger.error(f"[FAILURE] All chunks failed for {filepath}.")
        from src.file_utils import prune_chunked_dir
        chunked_dir = os.path.join(os.path.dirname(filepath), '_chunked')
        prune_chunked_dir(chunked_dir)
        return None

    jobs = _generate_chunk_job_objects(
        filepath, config, system_prompt_content, response_field, llm_model_name, api_key_prefix, tiktoken_encoding_func
    )
    if not jobs:
        logger.info(f"No data loaded from {filepath}. Skipping.")
        return None
    halt_on_failure = config.get('halt_on_chunk_failure', True)
    completed_jobs = []
    rich_table = RichJobTable()
    failure_detected = False
    future_to_job = _submit_jobs(jobs, response_field)
    try:
        with Live(rich_table.build_table(jobs), console=rich_table.console, refresh_per_second=5) as live:
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                logger.info(f"Chunk {job.chunk_id_str} submitted")
                try:
                    result_job = future.result()
                except Exception as exc:
                    result_job = job
                    result_job.status = "failed"
                    result_job.error_message = str(exc)
                _handle_job_result(job, result_job, completed_jobs)
                live.update(rich_table.build_table(jobs))
                if result_job.status == "failed" and halt_on_failure:
                    logger.error(f"[HALT] Failure detected in chunk {result_job.chunk_id_str}. Halting remaining jobs.")
                    failure_detected = True
                    break
            if failure_detected:
                for fut in future_to_job:
                    if not fut.done():
                        fut.cancel()
        logger.info("All chunks completed. Aggregating results...")
        return _aggregate_and_cleanup(completed_jobs, filepath)
    finally:
        from rich_display import print_summary_table
        print_summary_table(jobs)


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
    # --- Output File Naming Convention ---
    # For test inputs (filepath under tests/input/), outputs are written to tests/output/.
    # For production, outputs go to output/.
    # Output file names:
    #   - Legacy: <basename>_results.csv
    #   - Forced chunking: <basename>_forced_results.csv
    #   - Otherwise: <basename>_results.csv
    # This ensures test runner can find files as expected.
    filename = os.path.basename(filepath)
    file_root, file_ext = os.path.splitext(filename)
    if 'tests' in filepath.replace('\\', '/').lower():
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        output_dir = os.path.join(project_root, 'tests', 'output')
    else:
        output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    config_force_chunk = config.get('force_chunk_count', 0)
    if 'legacy' in file_root.lower():
        out_suffix = '_results'
    elif config_force_chunk and config_force_chunk > 1:
        out_suffix = '_forced_results'
    else:
        out_suffix = '_results'
    output_filename = f"{file_root}{out_suffix}{file_ext}"
    output_path = os.path.join(output_dir, output_filename)
    if os.path.exists(output_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{file_root}{out_suffix}_{timestamp}{file_ext}"
        output_path = os.path.join(output_dir, output_filename)
        logger.info(f"Output file already exists. Using new filename: {output_filename}")

    logger.info(f"Processing file: {filepath}")
    df = None
    try:
        df = load_data(filepath)
        logger.info(f"Loaded {len(df)} rows from {filepath}")

        examples_dir = config.get('examples_dir')
        if not examples_dir:
            raise ValueError("'examples_dir' not found in config.yaml.")
        project_root = CONFIG_DIR.parent
        abs_examples_path = (project_root / examples_dir).resolve()
        if is_examples_file_default(abs_examples_path):
            system_prompt_content = load_prompt_template("batch_evaluation_prompt_generic")
        else:
            if not abs_examples_path.exists():
                raise FileNotFoundError(f"Examples file not found: {abs_examples_path}. Please provide a valid examples file or update your config.")
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
            model_name = config.get('openai_model_name', 'gpt-4o-mini-2024-07-18')
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except KeyError:
                logger.warning(f"[WARN] tiktoken does not recognize model '{model_name}', using cl100k_base encoding.")
                enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            logger.error("\n\n############################")
            logger.error("ERROR: You need tiktoken or half of the functionality explodes, my guy!")
            logger.error("RERUN: uv pip install -r requirements.txt")
            logger.error("############################\n\n")
            raise RuntimeError("You need tiktoken or half of the functionality explodes my guy. Run 'uv pip install -r requirements.txt'")

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
        logger.info(f"[SUBMITTED TOKENS] Total: {total_tokens}, Avg: {avg_tokens:.1f}, Max: {max_tokens}")

        if total_tokens > TOKEN_LIMIT:
            logger.error(f"[ERROR] Total submitted tokens ({total_tokens}) exceeds the allowed cap of {TOKEN_LIMIT} for a single batch.")
            logger.error("Please reduce your batch size or check your usage at https://platform.openai.com/usage.")
            logger.error("Batch submission halted. No API calls were made.")
            try:
                llm_client = LLMClient()
                update_token_log(llm_client.api_key, 0)
            except Exception as e:
                logger.error(f"[WARN] Could not log token usage: {e}")
            return False
        try:
            llm_client = LLMClient()
            update_token_log(llm_client.api_key, int(total_tokens))
        except Exception as e:
            logger.error(f"[WARN] Could not log token usage: {e}")

        if df.empty:
            logger.info(f"No data loaded from {filepath}. Skipping.")
            return False

        MAX_BATCH_SIZE = 50000
        if len(df) > MAX_BATCH_SIZE:
            logger.warning(f"[WARN] Input file contains {len(df)} rows. Only the first {MAX_BATCH_SIZE} will be sent to the API (limit is 50,000 per batch). The rest will be ignored for this run. Simultaneous requests to the API are not supported yet but I am working on it.")
            df = df.iloc[:MAX_BATCH_SIZE].copy() 

        force_chunk_count = config.get('force_chunk_count', 0)
        if force_chunk_count > 1 or config.get('split_token_limit', 500_000) < total_tokens:
            df_with_results = process_file_concurrently(
                filepath, config, system_prompt_content, RESPONSE_FIELD, model_name, llm_client.api_key, enc
            )
        else:
            llm_client = LLMClient()
            try:
                df_with_results = llm_client.run_batch_job(
                    df, system_prompt_content, response_field_name=RESPONSE_FIELD, base_filename_for_tagging=filename
                )
            except Exception as batch_exc:
                logger.error(f"[ERROR] Batch job failed for {filepath}: {batch_exc}")
                return False

        save_data(df_with_results.drop(columns=['custom_id'], errors='ignore'), output_path)
        logger.success(f"Processed {filepath}. Results saved to {output_path}")
        logger.success(f"Total rows successfully processed: {len(df_with_results)}")
        error_rows = df_with_results['llm_score'].str.contains('Error', case=False)
        if error_rows.any():
            logger.info(f"Total rows with errors: {error_rows.sum()}")
            if error_rows.sum() == len(df_with_results):
                logger.error(f"[BATCH FAILURE] All rows failed for {filepath}. Halting further processing.")
                return False
        
        try:
            input_col_name = 'input_tokens'
            output_col_name = 'output_tokens'
            if enc is None and (input_col_name not in df_with_results.columns or output_col_name not in df_with_results.columns):
                logger.error("\n\n############################")
                logger.error("ERROR: You need tiktoken or half of the functionality explodes, my guy!")
                logger.error("RERUN: uv pip install -r requirements.txt")
                logger.error("############################\n\n")
                raise RuntimeError("You need tiktoken or half of the functionality explodes my guy. Run 'uv pip install -r requirements.txt'")
            if enc is not None and (input_col_name not in df_with_results.columns or output_col_name not in df_with_results.columns):
                df_with_results[input_col_name] = df_with_results.apply(lambda row: count_input_tokens_per_row(row, system_prompt_content, RESPONSE_FIELD, enc), axis=1)
                df_with_results[output_col_name] = df_with_results.apply(lambda row: count_completion_tokens(row, enc), axis=1)
            n_input_tokens = int(df_with_results[input_col_name].sum()) if input_col_name in df_with_results.columns else 0
            n_output_tokens = int(df_with_results[output_col_name].sum()) if output_col_name in df_with_results.columns else 0
            model_name = config.get('openai_model_name', 'gpt-4o-2024-08-06')
            try:
                cost = CostEstimator.estimate_cost(model_name, n_input_tokens, n_output_tokens)
                logger.info(f"Estimated LLM cost: ${cost:.4f} (input: {n_input_tokens} tokens, output: {n_output_tokens} tokens, model: {model_name})")
            except Exception as ce:
                logger.error(f"Could not estimate cost: {ce}")
            try:
                log_token_usage_event(
                    api_key=llm_client.api_key,
                    model=model_name,
                    input_tokens=n_input_tokens,
                    output_tokens=n_output_tokens,
                    timestamp=None,
                    request_id=None
                )
                logger.info("Token usage event logged to output/token_usage_events.jsonl.")
            except Exception as log_exc:
                logger.error(f"[Token Logging Error] Failed to log token usage event: {log_exc}")
        except Exception as cost_exception:
            logger.error(f"[Cost Estimation Error] {cost_exception}")

    except Exception as e:
        logger.error(f"An error occurred while processing {filepath}: {e}")
        log_basename = os.path.splitext(filename)[0] + ".log"
        log_base_path = os.path.join(OUTPUT_DIR, log_basename)
        if os.path.exists(log_base_path):
            file_root, file_ext = os.path.splitext(log_basename)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_log_filename = f"{file_root}_{timestamp}{file_ext}"
            log_path = os.path.join(OUTPUT_DIR, new_log_filename)
            logger.info(f"Log file already exists. Using new filename: {new_log_filename}")
        else:
            log_path = log_base_path
        try:
            with open(log_path, 'w') as f_log:
                f_log.write(f"Failed to process {filepath}.\nError: {e}\n")
            logger.info(f"Logged error to {log_path}")
        except Exception as save_err:
            logger.error(f"Double fail: exception in processing and then in saving error log for {filepath}: {save_err} ... aborting.")
        return False

    return True

def get_request_mode(args):
    """
    Indicates in the CLI whether or not requests are being submitted or we're in a safe count/split mode.
    """
    if getattr(args, 'count_tokens', False) or getattr(args, 'split_tokens', False):
        return "Split/Count (NO REQUESTS MADE)"
    return "API Request/Batch"

def print_token_cost_stats():
    """
    Prints token/cost usage stats (all time, today, per-model breakdown) using token_tracker utilities.
    """
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')
    logger.info("\n================= TOKEN USAGE & COST STATS =================")
    logger.info("ALL TIME:")
    summary_all = get_token_usage_summary()
    print_token_cost_summary(summary_all)
    logger.info("\nTODAY:")
    summary_today = get_token_usage_summary(start_date=today, end_date=today)
    print_token_cost_summary(summary_today)
    logger.info("===========================================================\n")

def print_token_cost_summary(summary):
    total_tokens = summary.get('total_tokens', 0)
    total_cost = summary.get('total_cost', 0.0)
    breakdown = summary.get('breakdown', {})
    logger.info(f"  Total tokens: {total_tokens:,}")
    logger.info(f"  Total cost: ${total_cost:,.6f}")
    if breakdown:
        logger.info("  Per-model breakdown:")
        logger.info("    Model           | Tokens      | Cost      | Count")
        logger.info("    --------------- | ----------- | --------- | -----")
        for model, stats in breakdown.items():
            tokens = stats.get('tokens', 0)
            cost = stats.get('cost', 0.0)
            count = stats.get('count', 0)
            logger.info(f"    {model:<15} | {tokens:>11,} | ${cost:>8,.4f} | {count:>5}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BatchGrader Runner")
    parser.add_argument('--log_dir', type=str, default=None, help='Directory for log files (default: output/logs or as set by test runner)')
    args, unknown = parser.parse_known_args()
    if args.log_dir:
        logger = BatchGraderLogger(log_dir=args.log_dir)
    parser = argparse.ArgumentParser(description="BatchGrader CLI: batch LLM evaluation, token counting, and input splitting.")
    parser.add_argument('--count-tokens', action='store_true', help='Count tokens in input file(s) and print stats.')
    parser.add_argument('--split-tokens', action='store_true', help='Split input file(s) into parts not exceeding the configured token limit.')
    parser.add_argument('--file', type=str, default=None, help='Only process the specified file in the input directory.')
    parser.add_argument('--config', type=str, default=None, help='Path to alternate config YAML file (default: config/config.yaml).')
    parser.add_argument('--costs', action='store_true', help='Show token/cost usage stats and exit.')
    parser.add_argument('--statistics', action='store_true', help='Show API usage stats even in count/split modes.')

    if '--file' in sys.argv:
        try:
            file_arg_index = sys.argv.index('--file')
            if file_arg_index + 1 >= len(sys.argv) or sys.argv[file_arg_index + 1].startswith('--'):
                logger.error("usage: batch_runner.py [-h] [--count-tokens] [--split-tokens] [--file FILE] [--costs] [--statistics]")
                logger.error("batch_runner.py: error: argument --file: expected one argument (the filename). It must be placed immediately after --file.")
                logger.error("Example: python batch_runner.py --file my_data.csv")
                sys.exit(2)
        except ValueError: 
            logger.error("severe oof error: basically something is COOKED if this happens") # passes to let us fail naturally as god intended because it's over
            pass

    args = parser.parse_args()
    config = load_config(args.config)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    INPUT_DIR = str(PROJECT_ROOT / config['input_dir'])
    OUTPUT_DIR = str(PROJECT_ROOT / config['output_dir'])
    RESPONSE_FIELD = config['response_field']
    TOKEN_LIMIT = config.get('token_limit', 2_000_000)
    split_token_limit = config.get('split_token_limit', 500_000)
    model_name = config['openai_model_name']

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    show_stats = args.statistics or (not args.count_tokens and not args.split_tokens)

    llm_client = LLMClient()
    if not llm_client.api_key:
        logger.error("Error: OPENAI_API_KEY not set in config/config.yaml.")
        exit(1)
    else:
        tokens_today = get_token_usage_for_day(llm_client.api_key)
        if show_stats:
            logger.info("\n================= API USAGE =================")
            logger.info(f"TOTAL TOKENS SUBMITTED TODAY: {tokens_today:,}")
            logger.info(f"TOKEN LIMIT: {TOKEN_LIMIT}")
            logger.info(f"TOKENS REMAINING: {TOKEN_LIMIT - tokens_today:,}")
            logger.info(f"SPLIT TOKEN LIMIT: {split_token_limit}")
            logger.info(f"System is running in {get_request_mode(args)} mode.")
            logger.info("==============================================\n")

    if getattr(args, 'costs', False):
        print_token_cost_stats()
        exit(0)

    logger.info(f"Valid INPUT_DIR: {INPUT_DIR}")
    logger.info(f"Valid OUTPUT_DIR: {OUTPUT_DIR}")

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
            logger.error(f"File {file_arg} not found at {resolved_path}.")
            logger.error("Halting: Missing input file.")
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
        logger.info(f"Nothing found in {INPUT_DIR} (looked for .csv, .json, .jsonl, if your data isn't in one of these formats please reformat.)")
        exit(0)

    def get_token_counter(system_prompt_content, response_field, enc):
        def count_submitted_tokens(row):
            sys_tokens = len(enc.encode(system_prompt_content))
            user_prompt = f"Please evaluate the following text: {str(row[response_field])}"
            user_tokens = len(enc.encode(user_prompt))
            return sys_tokens + user_tokens
        return count_submitted_tokens

    for resolved_path, df in files_found:
        try:
            logger.info(f"\nProcessing file: {resolved_path}")
            
            examples_dir = config.get('examples_dir')
            if not examples_dir:
                raise ValueError("'examples_dir' not found in config.yaml.")
            project_root = CONFIG_DIR.parent
            abs_examples_path = (project_root / examples_dir).resolve()
            if is_examples_file_default(abs_examples_path):
                system_prompt_content = load_prompt_template("batch_evaluation_prompt_generic")
            else:
                if not abs_examples_path.exists():
                    raise FileNotFoundError(f"Examples file not found: {abs_examples_path}. Please provide a valid examples file or update your config.")
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
                logger.error("\n\n############################")
                logger.error("ERROR: You need tiktoken or half of the functionality explodes, my guy!")
                logger.error("RERUN: uv pip install -r requirements.txt")
                logger.error("############################\n\n")
                raise RuntimeError("You need tiktoken or half of the functionality explodes my guy. Run 'uv pip install -r requirements.txt'")

            if args.count_tokens or args.split_tokens:
                if enc is None:
                    logger.error("\n\n############################")
                    logger.error("ERROR: You need tiktoken or half of the functionality explodes, my guy!")
                    logger.error("RERUN: uv pip install -r requirements.txt")
                    logger.error("############################\n\n")
                    raise RuntimeError("You need tiktoken or half of the functionality explodes my guy. Run 'uv pip install -r requirements.txt'")
                token_counter = get_token_counter(system_prompt_content, RESPONSE_FIELD, enc)
                token_counts = df.apply(token_counter, axis=1)
                total_tokens = token_counts.sum()
                avg_tokens = token_counts.mean()
                max_tokens = token_counts.max()
                logger.info(f"[TOKEN COUNT] Total: {total_tokens}, Avg: {avg_tokens:.1f}, Max: {max_tokens}")
                if args.split_tokens:
                    display_name = os.path.basename(resolved_path)
                    if total_tokens <= TOKEN_LIMIT:
                        logger.info(f"File {display_name} does not exceed the token limit. No split needed.")
                    else:
                        logger.info(f"Splitting {display_name} into chunks not exceeding {split_token_limit} tokens...")
                        output_files, token_counts = split_file_by_token_limit(resolved_path, split_token_limit, token_counter, RESPONSE_FIELD, output_dir=INPUT_DIR)
                        logger.info(f"Split complete. Output files: {output_files}")
                        for out_file, tok_count in zip(output_files, token_counts):
                            logger.info(f"Output file: {out_file} | Tokens: {tok_count}")
                continue
            # Fastest fix I ever done did (2025-05-11). This prevents submitting further batches if one fails. Halts immediately. RIP my poor API credits.
            ok = process_file(resolved_path)
            if not ok:
                logger.error(f"[BATCH HALTED] Halting further batch processing due to failure in {resolved_path}.")
                break
        except Exception as e:
            logger.error(f"[BATCH HALTED] Error processing {resolved_path}: {e}")
            logger.error("Halting further batch processing due to failure.")
            break
    logger.success("Batch finished processing.\n")

    for handler in getattr(logger, 'file_logger', logging.getLogger()).handlers:
        try:
            handler.flush()
        except Exception:
            pass
        try:
            handler.close()
        except Exception:
            pass
    print("[CLEANUP] Logger handlers flushed and closed.")

    if show_stats:
        print_token_cost_stats()
