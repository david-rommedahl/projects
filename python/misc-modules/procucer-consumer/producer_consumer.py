"""
Hybrid Thread-Process Producer-Consumer Queue Implementation

This module implements a flexible producer-consumer pattern using a hybrid approach:
- An I/O-bound producer running in a thread
- CPU-bound consumers running in separate processes
- Communication through a shared queue

The implementation includes:
- A class-based producer-consumer architecture
- Support for Pydantic models for data validation
- Back-pressure handling through a size-limited queue
- Multiple consumers for parallel processing
- Extensible design through inheritance
- Automatic worker scaling based on available CPU cores
- Configurable execution modes (threads vs processes)

This pattern is useful for processing large datasets where:
- The producer is primarily I/O-bound (reading files, API calls)
- The consumers are CPU-bound (processing data)
- Processing is independent and can benefit from parallelization

Usage:
    # Create producer and consumer instances
    producer = JsonlProducer("data.jsonl", model_class=DataItem)
    consumer = Consumer()

    # Create and run the system with default modes
    system = ProducerConsumerSystem(producer=producer, consumer=consumer)
    system.run()

    # Or customize execution modes and worker counts
    system = ProducerConsumerSystem(
        producer=producer,
        consumer=consumer,
        producer_mode=ExecutionMode.THREAD,
        consumer_mode=ExecutionMode.PROCESS
    )
    system.run(max_workers=4, max_queue_size=8)
"""

import json
import logging
import logging.config
import logging.handlers
from abc import ABC, abstractmethod
from enum import Enum
from multiprocessing import Process, Queue, cpu_count, current_process
from multiprocessing.queues import Queue as QueueType
from pathlib import Path
from threading import Thread
from typing import Any, Iterator


# Define execution modes
class ExecutionMode(Enum):
    """Execution mode for producers and consumers."""

    THREAD = "thread"
    PROCESS = "process"


# Create logger but don't configure it yet
logger = logging.getLogger(__name__)


def write_jsonl(
    path: Path,
    json_objects: list[dict[str, Any]] | dict[str, Any],
    overwrite: bool = False,
) -> None:
    """Helper function which writes JSON objects to a `.jsonl` file.

    This function writes JSON objects to a JSONL (JSON Lines) file, where each line contains a valid JSON object.
    If the file already exists and overwrite is False, the existing file will be renamed with '_old' suffix.

    Args:
        path: Path object pointing to where the JSONL file should be written.
        json_objects: Either a list of dictionaries or a dictionary to write to the file.
            - If a list of dictionaries, each dictionary is written as a separate line.
            - If a dictionary, each key-value pair is written as a separate line in the format {"key": value}.
        overwrite: If True, adds to the existing file. If False, rename the existing file. Defaults to False.
    """
    # If the path already exists, rename the old path.
    if path.exists() and not overwrite:
        path.rename(path.with_stem(f"{path.stem}_old"))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        if isinstance(json_objects, list) and all(
            isinstance(x, dict) for x in json_objects
        ):
            for d in json_objects:
                json.dump(d, f)
                f.write("\n")
        elif isinstance(json_objects, dict):
            for k, v in json_objects.items():
                json.dump({k: v}, f)
                f.write("\n")


def detect_logging_config() -> dict | None:
    """Auto-detect current logging configuration for child processes.

    Returns:
        dict: dictConfig-compatible logging configuration if multiprocessing-safe logging is detected, None otherwise
    """
    root_logger = logging.getLogger()
    queue_handler = None

    # Check if we have a QueueHandler in the root logger
    for handler in root_logger.handlers:
        if isinstance(handler, logging.handlers.QueueHandler):
            queue_handler = handler
            break

    if not queue_handler:
        return None

    # Extract format from handler
    log_format = (
        getattr(queue_handler.formatter, "_fmt", None)
        if queue_handler.formatter
        else "%(asctime)s | %(name)s | %(levelname)s | PID: %(process)d | Thread: %(thread)d | Line: %(lineno)s | %(message)s"
    )

    # Build loggers configuration for all configured loggers
    loggers = {}
    for name, logger_obj in logging.Logger.manager.loggerDict.items():
        if (
            isinstance(logger_obj, logging.Logger)
            and logger_obj.level != logging.NOTSET
        ):
            loggers[name] = {
                "level": logger_obj.level,
                "propagate": logger_obj.propagate,
            }

    # Generate dictConfig-compatible configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": {"format": log_format}},
        "handlers": {
            "queue": {
                "class": "logging.handlers.QueueHandler",
                "queue": queue_handler.queue,
                "formatter": "default",
            }
        },
        "root": {"level": root_logger.level, "handlers": ["queue"]},
    }

    if loggers:
        config["loggers"] = loggers

    return config


class Producer(ABC):
    """Base producer class that handles queue operations and consumer management."""

    def __init__(self, execution_mode: ExecutionMode = ExecutionMode.THREAD) -> None:
        """Initialize the producer.

        Args:
            execution_mode: Whether to run producer in a thread or process
        """
        self.execution_mode = execution_mode

    def produce(
        self, queue: QueueType, num_consumers: int, logging_config: dict | None = None
    ):
        """Process items from the generator and put them in the queue.

        Args:
            queue: Queue to put items into
            num_consumers: Number of consumer processes
            logging_config: Optional logging configuration for process-safe logging
        """
        # Configure logging for this process if running as a process
        if self.execution_mode == ExecutionMode.PROCESS and logging_config:
            logging.config.dictConfig(logging_config)

        for item in self._generate_items():
            # Check if queue is full before putting
            if queue.full():
                logger.debug("Queue is full, producer waiting...")
            # Put the item in the queue
            queue.put(item)
            logger.debug("Produced item %s", item)

        # Send sentinel value to signal end of production for each consumer
        for _ in range(num_consumers):
            if queue.full():
                logger.debug("Queue is full, producer waiting to send sentinel...")
            queue.put(None)
        logger.debug("Producer finished")

    @abstractmethod
    def _generate_items(self, *args, **kwargs) -> Iterator[Any]:
        """Generate items to be put in the queue."""
        raise NotImplementedError("Subclasses must implement _generate_items")


class BaseResultProcessor(ABC):
    """Base class for processing results from a queue.

    This abstract class provides a common interface for processing results
    from a queue, with progress bar updates.
    """

    def __init__(
        self,
        total_items: int = 0,
        description: str = "Processing",
        execution_mode: ExecutionMode = ExecutionMode.THREAD,
    ):
        """Initialize the base result processor.

        Args:
            total_items: Total number of items to process. Used for displaying progress
            description: Description for progress messages
            execution_mode: Whether to run result processor in a thread or process
        """
        self.total_items = total_items
        self.description = description
        self.total_processed = 0
        self.execution_mode = execution_mode

    def process_results(
        self, result_queue: QueueType, logging_config: dict | None = None
    ):
        """Process results from the queue.

        This method continuously gets results from the queue and processes them
        until a sentinel value (None) is received.

        Args:
            result_queue: Queue to receive results from consumer processes
            logging_config: Optional logging configuration for process-safe logging
        """
        # Configure logging for this process if running as a process
        if self.execution_mode == ExecutionMode.PROCESS and logging_config:
            logging.config.dictConfig(logging_config)

        while True:
            # Get batch of results from queue
            results = result_queue.get()

            # None is the sentinel value indicating end of processing
            if results is None:
                self._on_completion()
                break

            # Process the results
            processed_count = self._process_result(results)

            # Update progress and log
            if processed_count > 0:
                self.total_processed += processed_count
                self._log_progress()

    def _log_progress(self):
        """Log the current progress."""
        if self.total_items > 0:
            bar_width = 30
            progress = self.total_processed / self.total_items
            progress_percent = int(progress * 100)
            filled_chars = int(progress * bar_width)
            progress_bar = f"[{'=' * filled_chars}{'>' if progress_percent < 100 else ''}{'.' * (bar_width - filled_chars)}]"
            logger.info(
                "%s: %d%% %s %d/%d",
                self.description,
                progress_percent,
                progress_bar,
                self.total_processed,
                self.total_items,
            )
        else:
            logger.info(
                "%s: %d items processed", self.description, self.total_processed
            )

    @abstractmethod
    def _process_result(self, results: Any) -> int:
        """Process a result from the queue.

        Args:
            results: Any result object from the queue

        Returns:
            Number of items processed
        """
        pass

    def _on_completion(self):
        """Called when processing is complete."""
        logger.info(
            "Result processor finished, processed %d items", self.total_processed
        )


class Consumer:
    """Consumer that processes items from a queue."""

    def __init__(self, execution_mode: ExecutionMode = ExecutionMode.PROCESS) -> None:
        """Initialize the consumer.

        Args:
            execution_mode: Whether to run consumer in a thread or process
        """
        self.execution_mode = execution_mode

    def consume(
        self,
        queue: QueueType,
        result_queue: QueueType | None = None,
        logging_config: dict | None = None,
    ):
        """Continuously get and process items from the queue.

        Args:
            queue: Queue to get items from
            result_queue: Optional queue to send results to
            logging_config: Optional logging configuration for process-safe logging
        """
        # Configure logging for this process if running as a process
        if self.execution_mode == ExecutionMode.PROCESS and logging_config:
            logging.config.dictConfig(logging_config)

        process_name = current_process().name
        logger.debug("Consumer %s started", process_name)

        while True:
            # Get item from queue
            item = queue.get()
            # Check for sentinel value
            if item is None:
                logger.debug(
                    "Consumer %s received sentinel value, exiting", process_name
                )
                break
            # Process the item (simulate CPU-heavy task)
            self._process_item(item, result_queue)

        logger.debug("Consumer %s finished", process_name)

    def _process_item(self, item, result_queue: QueueType | None = None):
        """Process a single item from the queue.

        This method is called for each item retrieved from the queue.
        In this base implementation, it simply logs the item. Subclasses
        should override this method to implement specific processing logic.

        Args:
            item: The item to process, typically a Pydantic model instance
                 validated by the producer
            result_queue: Optional queue to send results to
        """
        process_name = current_process().name
        logger.debug("Consumer %s consumed: %s", process_name, item)


class ProducerConsumerSystem:
    """Orchestrates the producer-consumer pattern with configurable workers."""

    def __init__(
        self,
        producer: Producer,
        consumer: Consumer,
        result_processor: BaseResultProcessor | None = None,
    ):
        """Initialize the system with producer and consumer instances.

        Args:
            producer: Producer instance to use for producing items
            consumer: Consumer instance to use for consuming items
            result_processor: Optional processor for handling results from consumers
        """
        self.producer = producer
        self.consumer = consumer
        self.result_processor = result_processor

    def _calculate_num_workers(self, max_workers: int | None = None) -> int:
        """Calculate the optimal number of workers based on execution modes.

        This method determines the ideal number of worker processes or threads
        based on the available CPU cores and the chosen execution modes for
        producers and consumers.

        Args:
            max_workers: Optional upper limit on the number of workers to create.
                If provided, the method will return min(calculated_workers, max_workers).

        Returns:
            int: The optimal number of worker processes or threads to create
        """
        total_cpus = cpu_count()
        process_count = 0

        # Calculate optimal worker count based on execution modes
        # Count processes that need dedicated CPUs
        if self.producer.execution_mode == ExecutionMode.PROCESS:
            process_count += 1

        # Count result processor if it's a process
        if (
            self.result_processor
            and self.result_processor.execution_mode == ExecutionMode.PROCESS
        ):
            process_count += 1

        if self.consumer.execution_mode == ExecutionMode.PROCESS:
            # Use remaining CPUs for consumer processes
            # Each process gets its own CPU core
            num_workers = max(1, total_cpus - process_count)
        else:
            # For thread consumers, they all share a single process
            # Due to Python's GIL, CPU-bound threads won't run in parallel
            # But for I/O-bound work, we can use more threads than CPUs
            # Count thread components that share the GIL
            thread_count = 1  # Main thread
            if self.producer.execution_mode == ExecutionMode.THREAD:
                thread_count += 1
            if (
                self.result_processor
                and self.result_processor.execution_mode == ExecutionMode.THREAD
            ):
                thread_count += 1

            if process_count > 0:
                # If any component is a process, threads share the main process
                num_workers = total_cpus * 2
            else:
                # If all components are threads, they all share the main process
                # and are subject to the GIL
                num_workers = total_cpus * 2 - thread_count

        return num_workers if not max_workers else min(num_workers, max_workers)

    def run(
        self,
        max_workers: int | None = None,
        max_queue_size: int | None = None,
    ):
        """Run the producer-consumer system.

        This method starts the producer and consumer workers according to the configured
        execution modes. It handles the creation of processes or threads, sets up the
        shared queue, and manages the lifecycle of all workers.

        Args:
            max_workers: Optional upper limit on the number of consumer workers.
                If not provided, an optimal number will be calculated based on CPU count.
            max_queue_size: Optional maximum size for the shared queue.
                If not provided, defaults to 2 * number of workers.
                This parameter helps control memory usage and back-pressure.

        Note:
            This method blocks until all workers have completed their tasks.
            The producer will finish first, followed by the consumers once they
            have processed all items in the queue.
        """
        # Calculate optimal number of workers
        num_workers = self._calculate_num_workers(max_workers)
        max_size = max_queue_size if max_queue_size else num_workers * 2

        # Auto-detect logging configuration
        logging_config = detect_logging_config()

        # Log the system configuration
        if self.result_processor:
            logger.info(
                "Using %d %s consumers + 1 %s producer + 1 %s result processor",
                num_workers,
                self.consumer.execution_mode.value,
                self.producer.execution_mode.value,
                self.result_processor.execution_mode.value,
            )
        else:
            logger.info(
                "Using %d %s consumers + 1 %s producer",
                num_workers,
                self.consumer.execution_mode.value,
                self.producer.execution_mode.value,
            )

        # Create a queue with limited size
        queue: Queue = Queue(maxsize=max_size)

        # Create result queue if result processor is provided
        result_queue: Queue | None = None
        result_worker: Thread | Process | None = None
        if self.result_processor:
            result_queue = Queue()
            if self.result_processor.execution_mode == ExecutionMode.PROCESS:
                result_worker = Process(
                    target=self.result_processor.process_results,
                    args=(result_queue, logging_config),
                    name="ResultProcessor",
                )
            else:  # THREAD mode
                result_worker = Thread(
                    target=self.result_processor.process_results,
                    args=(result_queue, logging_config),
                    name="ResultProcessor",
                )
            result_worker.start()

        # Create consumer processes or threads
        consumer_workers = []
        worker: Process | Thread
        for i in range(num_workers):
            if self.consumer.execution_mode == ExecutionMode.PROCESS:
                worker = Process(
                    target=self.consumer.consume,
                    args=(queue, result_queue, logging_config),
                    name=f"Consumer-{i + 1}",
                )
            else:  # THREAD mode
                worker = Thread(
                    target=self.consumer.consume,
                    args=(queue, result_queue, logging_config),
                    name=f"Consumer-{i + 1}",
                )

            consumer_workers.append(worker)
            worker.start()

        producer_worker: Process | Thread
        # Start producer in a thread or process
        if self.producer.execution_mode == ExecutionMode.PROCESS:
            producer_worker = Process(
                target=self.producer.produce,
                args=(queue, num_workers, logging_config),
                name="Producer",
            )
        else:  # THREAD mode
            producer_worker = Thread(
                target=self.producer.produce,
                args=(queue, num_workers, logging_config),
                name="Producer",
            )

        producer_worker.start()

        # Wait for producer to complete
        producer_worker.join()
        logger.debug("Producer completed")

        # Wait for all consumers to complete
        for worker in consumer_workers:
            worker.join()

        # Signal result processor to finish if provided
        if self.result_processor and result_queue:
            assert result_worker is not None
            result_queue.put(None)
            result_worker.join()

        logger.debug("All workers completed")
