import json
import os
import tempfile
import time
import uuid
from datetime import datetime

import openai
from openai import OpenAI
from rich.console import Console

from config_loader import load_config
from logger import logger as global_logger_instance

config = load_config()


def get_default_api_key():
    return config['openai_api_key']


def get_default_model():
    return config['openai_model_name']


def get_default_endpoint():
    return config['batch_api_endpoint']


def get_default_max_tokens():
    return int(config['max_tokens_per_response'])


def get_default_poll_interval():
    return int(config['poll_interval_seconds'])


def get_default_response_field():
    return config['response_field']


class SimulatedChunkFailureError(Exception):
    pass


class LLMClient:

    def __init__(self, model=None, api_key=None, endpoint=None, logger=None):
        self.logger = logger or global_logger_instance
        self.api_key = api_key or get_default_api_key()
        self.model = model or get_default_model()
        self.endpoint = endpoint or get_default_endpoint()
        self.max_tokens = get_default_max_tokens()
        self.poll_interval = get_default_poll_interval()
        openai.api_key = self.api_key
        self.client = OpenAI(api_key=self.api_key)
        self.logger.info(
            f"LLMClient initialized. Model: {self.model}, Endpoint: {self.endpoint}"
        )

    def _prepare_batch_requests(self, df, system_prompt_content,
                                response_field_name):
        requests = []
        df['custom_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        for _, row in df.iterrows():
            custom_id = row['custom_id']
            text_to_evaluate = str(row[response_field_name])
            messages = [{
                "role": "system",
                "content": system_prompt_content
            }, {
                "role":
                "user",
                "content":
                f"Please evaluate the following text: {text_to_evaluate}"
            }]
            body = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens
            }
            requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": self.endpoint,
                "body": body
            })
        return requests, df

    def _upload_batch_input_file(self, requests_data,
                                 base_filename_for_tagging):
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w+",
                    delete=False,
                    suffix=".jsonl",
                    prefix=f"{base_filename_for_tagging}_") as tmp_f:
                temp_file_path = tmp_f.name
                for request_item in requests_data:
                    tmp_f.write(json.dumps(request_item) + "\n")
            with open(temp_file_path, "rb") as f_rb:
                batch_input_file = self.client.files.create(file=f_rb,
                                                            purpose="batch")
            return batch_input_file.id
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def _manage_batch_job(self, input_file_id, source_filename):
        console = Console()
        console.print(
            f"Creating batch job for {source_filename} with file ID: {input_file_id}"
        )
        batch_job = self.client.batches.create(
            input_file_id=input_file_id,
            endpoint=self.endpoint,
            completion_window="24h",
            metadata={"source_file": source_filename})
        last_status = None
        terminal_statuses = ["completed", "failed", "expired", "cancelled"]
        while True:
            retrieved_batch = self.client.batches.retrieve(batch_job.id)
            status = retrieved_batch.status
            status_line = f"[{datetime.now():%y/%m/%d %H:%M:%S}] INFO     Batch job {batch_job.id} status: {status}"
            if status not in terminal_statuses:
                if status == last_status:
                    console.print(status_line,
                                  end="\r",
                                  highlight=False,
                                  soft_wrap=True)
                else:
                    console.print(status_line)
            else:
                console.print(status_line)
                return retrieved_batch
            last_status = status
            time.sleep(self.poll_interval)

    def _llm_parse_batch_output_file(self, output_file_id):
        """Parses the batch output file and returns a map of custom_id to results."""
        item_results_map = {}
        try:
            file_response = self.client.files.content(output_file_id)
            output_data = file_response.text
            for line in output_data.strip().splitlines():
                item = json.loads(line)
                custom_id = item.get("custom_id")
                if not custom_id:
                    self.logger.warning(
                        f"Warning: Found item in output without custom_id: {item}"
                    )
                    continue

                if item.get("response") and not item.get("error"):
                    resp = item.get("response")
                    try:
                        content = resp["body"]["choices"][0]["message"][
                            "content"]
                        item_results_map[custom_id] = content
                    except (KeyError, IndexError, TypeError) as e_parse:
                        self.logger.warning(
                            f"Error parsing successful response for custom_id {custom_id}: {e_parse}. Full item: {item}"
                        )
                        item_results_map[
                            custom_id] = "Error: Malformed response data"
                elif (err := item.get("error")):
                    err_msg = err.get('message', 'Unknown error')
                    err_code = err.get('code', 'N/A')
                    item_results_map[
                        custom_id] = f"Error: {err_code} - {err_msg}"
                    self.logger.warning(
                        f"Request {custom_id} failed: {err_code} - {err_msg}")
                else:
                    item_results_map[
                        custom_id] = "Error: Unknown structure in output item"
                    self.logger.warning(
                        f"Warning: Unknown structure for item with custom_id {custom_id}: {item}"
                    )
            return item_results_map
        except Exception as e_file:
            self.logger.error(
                f"Critical error retrieving or parsing batch output file {output_file_id}: {e_file}",
                exc_info=True)

            raise IOError(
                f"Failed to retrieve or parse batch output file {output_file_id}"
            ) from e_file

    def _llm_retrieve_batch_error_file(self, error_file_id):
        """Retrieves and logs the content of a batch error file."""
        try:
            error_file_content = self.client.files.content(error_file_id).text
            self.logger.info(
                f"Batch Error File Content ({error_file_id}):\n{error_file_content[:1000]}..."
            )
        except Exception as e:
            self.logger.warning(
                f"Error retrieving batch error file {error_file_id}: {e}",
                exc_info=True)

    def _process_batch_outputs(self, batch_job_obj, df_with_custom_ids):
        results_map = {}
        llm_output_column_name = 'llm_score'

        output_file_id = getattr(batch_job_obj, 'output_file_id', None)
        error_file_id = getattr(batch_job_obj, 'error_file_id', None)

        if batch_job_obj.status == "completed":
            if output_file_id:
                self.logger.info(f"Retrieving output file: {output_file_id}")
                try:
                    parsed_item_results = self._llm_parse_batch_output_file(
                        output_file_id)
                    results_map.update(parsed_item_results)
                except IOError as e_parse_file:
                    self.logger.error(
                        f"Failed to process batch output file {output_file_id} due to: {e_parse_file}"
                    )

                    for cid in df_with_custom_ids['custom_id']:
                        results_map[
                            cid] = "Error: Failed to retrieve/parse batch output file"
            else:
                self.logger.warning(
                    "Batch completed, but no output file ID was provided.")
                for cid in df_with_custom_ids['custom_id']:
                    results_map[
                        cid] = "Error: Batch completed with no output file"

            if error_file_id:
                self.logger.info(f"Retrieving error file: {error_file_id}")
                self._llm_retrieve_batch_error_file(error_file_id)
        else:
            self.logger.warning(
                f"Batch job did not complete successfully. Status: {batch_job_obj.status}"
            )
            for cid in df_with_custom_ids['custom_id']:
                results_map[
                    cid] = f"Error: Batch job status - {batch_job_obj.status}"

        for cid in df_with_custom_ids['custom_id']:
            if cid not in results_map:
                self.logger.warning(
                    f"Custom ID {cid} not found in parsed batch results, marking as error."
                )
                results_map[cid] = "Error: Result not found in batch output"

        df_with_custom_ids[llm_output_column_name] = df_with_custom_ids[
            'custom_id'].map(results_map)

        df_with_custom_ids[llm_output_column_name] = df_with_custom_ids[
            llm_output_column_name].fillna(
                "Error: No result found for custom_id (fallback)")
        return df_with_custom_ids

    def run_batch_job(self,
                      df,
                      system_prompt_content,
                      response_field_name=None,
                      base_filename_for_tagging=None):
        """
        runs a full batch job: prepares requests, uploads input, manages job, processes results.
        Returns the processed DataFrame with results.
        """

        if self.api_key == "TEST_KEY_FAIL_CONTINUE":
            if base_filename_for_tagging in [
                    "chunk_with_failure_chunk_2", "chunk_with_failure_chunk_4"
            ]:
                self.logger.info(
                    f"Simulating failure for chunk: {base_filename_for_tagging} due to TEST_KEY_FAIL_CONTINUE"
                )
                raise SimulatedChunkFailureError(
                    f"Simulated failure for {base_filename_for_tagging}")

            self.logger.info(
                f"Simulating success for chunk: {base_filename_for_tagging} due to TEST_KEY_FAIL_CONTINUE (non-failing chunk)"
            )

            llm_output_column_name = config.get('llm_output_column_name',
                                                'llm_response')
            df_with_ids = df.copy()
            if 'custom_id' not in df_with_ids.columns:
                df_with_ids['custom_id'] = [
                    str(uuid.uuid4()) for _ in range(len(df_with_ids))
                ]
            df_with_ids[
                llm_output_column_name] = f"Mocked success for {base_filename_for_tagging}"
            return df_with_ids

        if response_field_name is None:
            response_field_name = get_default_response_field()
        if base_filename_for_tagging is None:
            base_filename_for_tagging = "batch_input"
        batch_requests_data, df_with_ids = self._prepare_batch_requests(
            df, system_prompt_content, response_field_name)
        if not batch_requests_data:
            self.logger.warning("No requests generated. Skipping.")
            return df
        input_file_id = self._upload_batch_input_file(
            batch_requests_data, base_filename_for_tagging)
        final_batch_obj = self._manage_batch_job(input_file_id,
                                                 base_filename_for_tagging)
        df_with_results = self._process_batch_outputs(final_batch_obj,
                                                      df_with_ids)
        return df_with_results
