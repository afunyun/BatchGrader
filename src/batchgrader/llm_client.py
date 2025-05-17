import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime
from typing import Any, Dict

import tenacity
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from rich.console import Console

# from logger import logger as global_logger_instance # DEPRECATED
from batchgrader.utils import get_encoder

logger = logging.getLogger(__name__)


def get_config_value(config: Dict[str, Any],
                     key: str,
                     default: Any = None) -> Any:
    """
    Helper function to safely extract values from config dict
    """
    return config.get(key, default)


class SimulatedChunkFailureError(Exception):
    pass


class LLMClient:

    def __init__(self, model=None, api_key=None, endpoint=None, config=None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Extract values from injected config or use provided parameters
        self.api_key = api_key or get_config_value(self.config,
                                                   "openai_api_key")
        self.model = model or get_config_value(self.config,
                                               "openai_model_name")
        self.endpoint = endpoint or get_config_value(self.config,
                                                     "batch_api_endpoint")
        self.max_tokens = int(
            get_config_value(self.config, "max_tokens_per_response", 1000))
        self.poll_interval = int(
            get_config_value(self.config, "poll_interval_seconds", 60))

        # Create client instance with the API key
        self.client = OpenAI(api_key=self.api_key)
        self.logger.info(
            f"LLMClient initialized. Model: {self.model}, Endpoint: {self.endpoint}"
        )

        # Retry settings initialization
        self.retry_settings = get_config_value(self.config, "retry_settings",
                                               {})
        self.max_retries = self.retry_settings.get("max_retries", 3)
        self.initial_backoff = self.retry_settings.get(
            "initial_backoff_seconds", 1)
        self.max_backoff = self.retry_settings.get("max_backoff_seconds", 60)

        self.retry_decorator = tenacity.retry(
            stop=tenacity.stop_after_attempt(self.max_retries),
            wait=tenacity.wait_exponential(multiplier=1,
                                           min=self.initial_backoff,
                                           max=self.max_backoff),
            retry=(tenacity.retry_if_exception_type(APIConnectionError)
                   | tenacity.retry_if_exception_type(RateLimitError)
                   | tenacity.retry_if_exception_type(APITimeoutError)
                   | tenacity.retry_if_exception(lambda e: isinstance(
                       e, APIStatusError) and e.status_code >= 500)),
            before_sleep=tenacity.before_sleep_log(self.logger,
                                                   logging.WARNING),
            reraise=True,
        )
        self.logger.info(
            f"LLMClient retry decorator configured: "
            f"max_retries={self.max_retries}, initial_backoff={self.initial_backoff}s, max_backoff={self.max_backoff}s"
        )

        # Initialize encoder using centralized utility
        self.encoder = get_encoder(self.model)

    def _prepare_batch_requests(self, df, system_prompt_content,
                                response_field_name):
        from .exceptions import FileFormatError
        requests = []
        if response_field_name not in df.columns:
            raise FileFormatError(
                f"Required column '{response_field_name}' not found in input DataFrame. "
                f"Columns present: {list(df.columns)}")
        df["custom_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
        for _, row in df.iterrows():
            custom_id = row["custom_id"]
            text_to_evaluate = str(row[response_field_name])
            messages = [
                {
                    "role": "system",
                    "content": system_prompt_content
                },
                {
                    "role":
                    "user",
                    "content":
                    f"Please evaluate the following text: {text_to_evaluate}",
                },
            ]
            body = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
            }
            requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": self.endpoint,
                "body": body,
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
                    prefix=f"{base_filename_for_tagging}_",
            ) as tmp_f:
                temp_file_path = tmp_f.name
                for request_item in requests_data:
                    tmp_f.write(json.dumps(request_item) + "\n")
            with open(temp_file_path, "rb") as f_rb:

                @self.retry_decorator
                def _do_upload():
                    self.logger.info(
                        f"Uploading batch input file: {temp_file_path} for {base_filename_for_tagging}"
                    )
                    return self.client.files.create(file=f_rb, purpose="batch")

                batch_input_file = _do_upload()
            self.logger.info(
                f"Successfully uploaded batch input file {batch_input_file.id} for {base_filename_for_tagging}"
            )
            return batch_input_file.id
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def _manage_batch_job(self, input_file_id, source_filename):
        console = Console()
        console.print(
            f"Creating batch job for {source_filename} with file ID: {input_file_id}"
        )

        @self.retry_decorator
        def _create_batch_job():
            self.logger.info(
                f"Attempting to create batch job for input_file_id: {input_file_id}"
            )
            return self.client.batches.create(
                input_file_id=input_file_id,
                endpoint=self.endpoint,
                completion_window="24h",
                metadata={"source_file": source_filename},
            )

        batch_job = _create_batch_job()
        self.logger.info(
            f"Batch job {batch_job.id} created successfully for {source_filename}."
        )

        last_status = None
        terminal_statuses = ["completed", "failed", "expired", "cancelled"]

        @self.retry_decorator  # Decorate the retrieve call
        def _retrieve_batch_status(job_id):
            self.logger.debug(
                f"Attempting to retrieve status for batch job: {job_id}")
            return self.client.batches.retrieve(job_id)

        while True:
            retrieved_batch = _retrieve_batch_status(
                batch_job.id)  # Call moved inside the main polling loop
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

            @self.retry_decorator
            def _get_file_content():
                self.logger.info(
                    f"Attempting to retrieve content for output file ID: {output_file_id}"
                )
                return self.client.files.content(output_file_id)

            file_response = _get_file_content()
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
                elif err := item.get("error"):
                    err_msg = err.get("message", "Unknown error")
                    err_code = err.get("code", "N/A")
                    item_results_map[
                        custom_id] = f"Error: {err_code} - {err_msg}"
                    self.logger.warning(
                        f"Request {custom_id} failed: {err_code} - {err_msg}")
                else:
                    item_results_map[custom_id] = (
                        "Error: Unknown structure in output item")
                    self.logger.warning(
                        f"Warning: Unknown structure for item with custom_id {custom_id}: {item}"
                    )
            return item_results_map
        except Exception as e_file:
            self.logger.error(
                f"Critical error retrieving or parsing batch output file {output_file_id}: {e_file}",
                exc_info=True,
            )

            raise IOError(
                f"Failed to retrieve or parse batch output file {output_file_id}"
            ) from e_file

    def _llm_retrieve_batch_error_file(self, error_file_id):
        """Retrieves and logs the content of a batch error file."""
        try:

            @self.retry_decorator
            def _get_error_file_content():
                self.logger.info(
                    f"Attempting to retrieve content for error file ID: {error_file_id}"
                )
                return self.client.files.content(error_file_id)

            error_file_response = _get_error_file_content()
            error_file_content = error_file_response.text
            self.logger.info(
                f"Batch Error File Content ({error_file_id}):\n{error_file_content[:1000]}..."
            )
        except Exception as e:
            self.logger.warning(
                f"Error retrieving batch error file {error_file_id}: {e}",
                exc_info=True)

    def _process_batch_outputs(self, batch_job_obj, df_with_custom_ids):
        results_map = {}
        llm_output_column_name = "llm_score"

        output_file_id = getattr(batch_job_obj, "output_file_id", None)
        error_file_id = getattr(batch_job_obj, "error_file_id", None)

        if batch_job_obj.status == "completed":
            if output_file_id:
                self.logger.info(f"Retrieving output file: {output_file_id}")
                try:
                    parsed_item_results = self._llm_parse_batch_output_file(
                        output_file_id)
                    results_map |= parsed_item_results
                except IOError as e_parse_file:
                    self.logger.error(
                        f"Failed to process batch output file {output_file_id} due to: {e_parse_file}"
                    )

                    for cid in df_with_custom_ids["custom_id"]:
                        results_map[cid] = (
                            "Error: Failed to retrieve/parse batch output file"
                        )
            else:
                self.logger.warning(
                    "Batch completed, but no output file ID was provided.")
                for cid in df_with_custom_ids["custom_id"]:
                    results_map[
                        cid] = "Error: Batch completed with no output file"

            if error_file_id:
                self.logger.info(f"Retrieving error file: {error_file_id}")
                self._llm_retrieve_batch_error_file(error_file_id)
        else:
            self.logger.warning(
                f"Batch job did not complete successfully. Status: {batch_job_obj.status}"
            )
            for cid in df_with_custom_ids["custom_id"]:
                results_map[
                    cid] = f"Error: Batch job status - {batch_job_obj.status}"

        for cid in df_with_custom_ids["custom_id"]:
            if cid not in results_map:
                self.logger.warning(
                    f"Custom ID {cid} not found in parsed batch results, marking as error."
                )
                results_map[cid] = "Error: Result not found in batch output"

        df_with_custom_ids[llm_output_column_name] = df_with_custom_ids[
            "custom_id"].map(results_map)

        df_with_custom_ids[llm_output_column_name] = df_with_custom_ids[
            llm_output_column_name].fillna(
                "Error: No result found for custom_id (fallback)")
        return df_with_custom_ids

    def run_batch_job(
        self,
        df,
        system_prompt_content,
        response_field_name=None,
        base_filename_for_tagging=None,
    ):
        """
        Run a batch job for a dataframe.

        Args:
            df: Pandas DataFrame with responses to evaluate
            system_prompt_content: System prompt for the model
            response_field_name: Column name in df containing text to evaluate
            base_filename_for_tagging: Base filename for temporary files

        Returns:
            DataFrame with evaluation results added
        """
        if response_field_name is None:
            response_field_name = get_config_value(self.config,
                                                   "response_field",
                                                   "response")

        if base_filename_for_tagging is None:
            base_filename_for_tagging = f"batch_job_{int(time.time())}"

        requests_data, df_with_custom_ids = self._prepare_batch_requests(
            df, system_prompt_content, response_field_name)

        input_file_id = self._upload_batch_input_file(
            requests_data, base_filename_for_tagging)

        batch_job_result = self._manage_batch_job(input_file_id,
                                                  base_filename_for_tagging)

        return self._process_batch_outputs(batch_job_result,
                                           df_with_custom_ids)
