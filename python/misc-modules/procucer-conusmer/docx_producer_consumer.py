"""Producer-consumer implementation for Docx document parsing."""

import logging
from itertools import batched
from multiprocessing.queues import Queue as QueueType
from pathlib import Path
from typing import Any, Iterable, Iterator

from producer_consumer import (
    BaseResultProcessor,
    Consumer,
    ExecutionMode,
    Producer,
    write_jsonl,
)

logger = logging.getLogger(__name__)


class DocxItemProducer(Producer):
    """Producer that generates batches of StandardMetadata objects for document parsing.

    This producer takes a metadata generator and produces batches of items for the consumer
    to process. It's designed to be I/O-bound, reading metadata and preparing it
    for processing by the consumers.
    """

    def __init__(
        self,
        item_gen: Iterable[Any],
        batch_size: int,
        execution_mode=ExecutionMode.THREAD,
    ):
        """Initialize the metadata producer.

        Args:
            item_gen: Generator or iterable providing items to parse
            batch_size: Number of items to include in each batch
            execution_mode: Whether to run producer in a thread or process
        """
        super().__init__(execution_mode)
        self.item_gen = item_gen
        self.batch_size = batch_size

    def _generate_items(self) -> Iterator[list[Any]]:
        """Generate batches of items from the provided generator.

        Yields:
            Batches of items for document parsing
        """

        for batch in batched(self.item_gen, self.batch_size):
            yield list(batch)


class DocxConsumer(Consumer):
    """Consumer that processes batches of Docx documents based on generated items.

    This consumer takes batches of items from the queue and processes the corresponding
    documents. It's designed to be CPU-bound, parsing documents and extracting content.
    """

    def __init__(
        self, parser, input_folder: Path, execution_mode=ExecutionMode.PROCESS
    ):
        """Initialize the document consumer.

        Args:
            parser: The parser instance that contains parsing logic
            input_folder: Base directory containing the input documents
            execution_mode: Whether to run consumer in a thread or process
        """
        super().__init__(execution_mode)
        self.parser = parser
        self.input_folder = input_folder
        self.successful_count = 0
        self.processed_count = 0

    def _process_item(self, batch: list[Any], result_queue: QueueType | None = None):
        """Process a batch of items by parsing the corresponding documents.

        Args:
            batch: List of items representing the documents to parse
        """
        successful = []
        failed = []
        batch_size = len(batch)

        for item in batch:
            try:
                # Parse the document using the parser's method
                result = self.parser.parse_single_document(item, self.input_folder)
                if result:
                    successful.append(result)
                    self.successful_count += 1
            except Exception as ex:
                # Handle parsing errors
                error_info = self.parser._handle_parsing_error(item, ex)
                failed.append(error_info)

        # Send both successful and failed results to the writer queue
        if result_queue:
            result_queue.put({"successful": successful, "failed": failed})

        # Track processed count
        self.processed_count += batch_size
        logger.debug("Processed %d documents so far", self.processed_count)


class ResultWriter(BaseResultProcessor):
    """Dedicated writer for handling file writing from multiple processes.

    This class runs in a separate thread or process and receives items from consumer processes
    through a queue, ensuring that only one process writes to the result file.
    It also updates the progress bar as results are processed and writes failed documents
    directly to a JSONL file.
    """

    def __init__(self, parser, total_items=0, execution_mode=ExecutionMode.THREAD):
        """Initialize the result writer.

        Args:
            parser: The parser instance that contains file writing logic
            total_items: Total number of items to process
            execution_mode: Whether to run result processor in a thread or process
        """
        super().__init__(total_items, "Parsing standards", execution_mode)
        self.parser = parser
        self.total_written = 0
        self.total_failed = 0

        # Create errors directory and file path
        self.error_file = (
            self.parser.output_folder
            / self.parser.execution_id
            / "errors"
            / "errors.jsonl"
        )
        self.error_file.parent.mkdir(parents=True, exist_ok=True)

    def _process_result(self, results: Any) -> int:
        """Process a result from the queue.

        Args:
            results: Result object from the queue

        Returns:
            Total number of items processed
        """
        # Handle dictionary results (backward compatibility)
        if isinstance(results, dict):
            # Handle successful documents
            successful = results.get("successful", [])
            if successful:
                self.parser._save_results(successful, overwrite=True)
                self.total_written += len(successful)

            # Handle failed documents - write directly to file
            failed = results.get("failed", [])
            if failed:
                self._write_failed_documents(failed)
                self.total_failed += len(failed)

            # Return total processed count
            return len(successful) + len(failed)

        # For other types of results, just return 1 as default
        return 1

    def _on_completion(self):
        """Called when processing is complete."""
        logger.debug(
            "Result writer finished, wrote %d documents, failed: %d",
            self.total_written,
            self.total_failed,
        )

    def _write_failed_documents(self, failed_documents):
        """Write failed documents directly to a JSONL file.

        Args:
            failed_documents: List of error information tuples to write
        """
        # Convert error tuples to dictionaries
        error_dicts = []
        for error_info in failed_documents:
            error_dicts.append(
                {
                    "document": error_info[0],
                    "error": error_info[1],
                    "traceback": error_info[2],
                }
            )

        # Use the existing write_jsonl function
        try:
            write_jsonl(self.error_file, error_dicts, overwrite=True)
        except Exception as ex:
            logger.error("Error writing failed documents to file: %s", ex)
