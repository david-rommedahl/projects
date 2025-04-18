import json
import warnings
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import floor
from typing import Any

import httpx
import tenacity
from tqdm import tqdm

from rag_data_uploader.uploaders.base import MAX_RETRIES, NUM_THREADS, RAGDocumentUploader
from rag_data_uploader.utils.exceptions import (
    BulkPostError,
    EnforceMappingError,
    GetMappingError,
    IndexCreateError,
    MappingNotProvidedError,
    PayloadTooLargeError,
)
from rag_data_uploader.utils.regexes import INDEX_REGEX

warnings.filterwarnings("ignore")


class ElasticsearchUploader(RAGDocumentUploader):
    """Class for uploading documents to an Elasticsearch instance."""

    def _create_index(self) -> None:
        """Method which creates an index in Elasticsearch/Opensearch."""
        if not self.mapping:
            raise MappingNotProvidedError("Mapping must be specified for non-existing indices")

        body = {"mappings": self.mapping}
        if self.alias:
            self._find_alias_indices()
            body.update({"aliases": {self.alias: {}}})
            self._enforce_alias_mapping()
        try:
            response = httpx.put(
                f"{self.url}/{self.index}",
                json=body,
                **self.req_kwargs,
            ).raise_for_status()
        except httpx.HTTPStatusError:
            raise IndexCreateError(f"Something went wrong with creating the index: {response.text}")

    @tenacity.retry(
        retry=tenacity.retry_if_not_exception_type(EnforceMappingError),
        wait=tenacity.wait_exponential_jitter(),
        stop=tenacity.stop_after_attempt(MAX_RETRIES),
        reraise=True,
    )
    def _enforce_alias_mapping(self) -> None:
        """Method which enforces the mapping in an alias which currently exists.

        It will raise an error if the keys of both mappings don't match.
        """
        try:
            response = httpx.get(
                f"{self.url}/{self.alias}/_mapping", **self.req_kwargs
            ).raise_for_status()
        except httpx.HTTPStatusError:
            return
        # If the alias already exists, enforce the mapping which is currently in the alias
        alias_mapping = next(iter(response.json().values())).get("mappings", {})

        if not alias_mapping["properties"].keys() == self.mapping["properties"].keys():
            print("User-defined mapping is in conflict with alias mapping.")
            print("Changing mapping to alias mapping.")
            self.mapping = alias_mapping

    @tenacity.retry(
        retry=tenacity.retry_if_not_exception_type((GetMappingError, KeyError)),
        wait=tenacity.wait_exponential_jitter(),
        stop=tenacity.stop_after_attempt(MAX_RETRIES),
        reraise=True,
    )
    def _get_mapping(self) -> dict[str, Any]:
        """Method which gets the mapping from an existing index.

        Uses the Elasticsearch mapping API.
        """
        alias = self.alias if self.alias else self.index
        try:
            response = httpx.get(
                f"{self.url}/{alias}/_mapping", **self.req_kwargs
            ).raise_for_status()
        except httpx.HTTPStatusError:
            raise GetMappingError(f"Could not get mapping for alias/index {alias}: {response.text}")

        response = response.json()
        mapping = next(iter(response.values())).get("mappings", {})

        if not mapping:
            raise KeyError(f"Could not extract mapping from response: {response}")
        print(f"Mapping has been changed to that of alias/index {alias}")
        return mapping

    def _bulk_batcher(self, documents: list[dict[str, Any]], batch_size: float) -> Generator[str]:
        """Helper method to create document batches.

        Converts documents to a string of the correct format
        to use the Elasticsearch/Opensearch bulk API and yields a batch of a predefined size.

        Args:
            documents:      A list of dictionary documents to be written to
                            Elasticsearch/Opensearch.
            batch_size:     A float indicating the maximum batch size in Mb.

        Yields:
            data:           A string which contains indexing actions followed by the
                            document to index, separated by a newline. The string must
                            be terminated by a newline.
        """

        action = {"index": {"_index": self.index}}
        # Convert batch size to batch_len, giving number or documents for a batch to stay within size limit
        num_docs = len(documents)
        max_size = round(
            (
                max([json.dumps(doc).__sizeof__() for doc in documents])
                + json.dumps(action).__sizeof__()
            )
            / 1e6,
            2,
        )
        batch_len = floor(batch_size / max_size) if max_size != 0 else num_docs
        for n in range(0, num_docs, batch_len):
            batch = documents[n : min(n + batch_len, num_docs)]
            data = "\n".join([f"{json.dumps(action)}\n{json.dumps(doc)}" for doc in batch]) + "\n"
            yield data.encode("utf-8")

    @tenacity.retry(
        retry=tenacity.retry_if_not_exception_type(PayloadTooLargeError),
        wait=tenacity.wait_exponential_jitter(),
        stop=tenacity.stop_after_attempt(MAX_RETRIES),
        reraise=True,
    )
    def _bulk_post_helper(self, data_batch: str) -> httpx.Response:
        """Helper method which uses the Elasticsearch/Opensearch bulk API for indexing documents.

        Args:
            data_batch:     An NDJSON string of documents to index in
                            Elasticsearch/Opensearch using the bulk API.
        """
        headers = {"content-type": "application/x-ndjson"}
        try:
            response = httpx.post(
                f"{self.url}/_bulk", data=data_batch, headers=headers, **self.req_kwargs
            ).raise_for_status()
        except httpx.HTTPStatusError as err:
            if err.response.status_code == 413:
                raise PayloadTooLargeError(
                    f"Data batch too large: {round(data_batch.__sizeof__()/1e6, 2)} Mb",
                )
            else:
                raise BulkPostError(err.response.text)
        return response

    def _upload_documents(
        self, documents: list[dict[str, Any]], batch_size: float, threading: bool
    ) -> None:
        """Method which uploads documents using bulk API.

        Uses threading and the Elasticsearch/Opensearch bulk API to index
        a list of JSON documents to an Elasticsearch/Opensearch document store.

         Args:
             documents:      A list of JSON documents to index.
             batch_size:     The batch size to index documents with, in Mb.
        """
        with tqdm(total=len(documents), desc="Uploading documents") as pbar:
            if threading:
                with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
                    futures = [
                        executor.submit(self._bulk_post_helper, data_batch)
                        for data_batch in self._bulk_batcher(documents, batch_size=batch_size)
                    ]
                    for future in as_completed(futures):
                        try:
                            response = future.result().raise_for_status()
                        except httpx.HTTPStatusError as err:
                            response = err.response
                            print(response.text)
                        self._get_errors(response)
                        pbar.update(len(response.json().get("items")))
            else:
                for data_batch in self._bulk_batcher(documents, batch_size=batch_size):
                    try:
                        response = self._bulk_post_helper(data_batch).raise_for_status()
                    except httpx.HTTPStatusError as err:
                        response = err.response
                        print(response.text)
                    self._get_errors(response)
                    pbar.update(len(response.json().get("items")))

    def _get_errors(self, response: httpx.Response) -> None:
        """Helper method which extracts errors from the response.

        Args:
            response:    Response object from Elasticsearch/Opensearch bulk API
        """

        response_json = response.json()
        if response_json.get("errors"):
            error_items = [
                item for item in response_json.get("items") if not item["index"].get("result")
            ]
            self.total_errors.extend(error_items)

    def _delete_index(self, index: str | list[str]) -> None:
        """Method used to delete an index from the document store.

        Args:
            index:    The index or list of indices that the user wants to delete.
        """

        if isinstance(index, list):
            index = ",".join(index)
        try:
            _ = httpx.delete(f"{self.url}/{index}", **self.req_kwargs).raise_for_status()
        except httpx.HTTPStatusError as err:
            raise httpx.HTTPStatusError(err.response.text)
        else:
            print(f"Index/indices {index} deleted")

    def _find_alias_indices(self, alias: str | None = None) -> None:
        """Finds indices for alias.

        Finds the indices which are associated with the current
        alias and match the current index name.

        Args:
            alias:    Name of the alias to find indices for.
        """

        if not alias:
            alias = self.alias
        index_stem = INDEX_REGEX.match(self.index)[0]
        try:
            response = httpx.get(f"{self.url}/_alias/{alias}", **self.req_kwargs).raise_for_status()
        except httpx.HTTPStatusError:
            return
        response_json = response.json()
        self.indices_to_replace = [
            index for index in response_json.keys() if index_stem == INDEX_REGEX.match(index)[0]
        ]

    def _remove_indices_from_alias(
        self, alias: str | None = None, indices: str | list[str] | None = None
    ) -> None:
        """Method which removes indices from the current alias.

        Args:
            alias:    Name of the alias to remove indices from.
            indices:  List of indices to remove.
        """

        if not alias:
            alias = self.alias
        if not indices:
            indices = self.indices_to_replace
        if isinstance(indices, str):
            indices = [indices]

        index_list = ",".join(indices)

        _ = httpx.delete(
            f"{self.url}/{index_list}/_alias/{alias}", **self.req_kwargs
        ).raise_for_status()
        print(f"Index/indices {index_list} removed from alias {alias}")
