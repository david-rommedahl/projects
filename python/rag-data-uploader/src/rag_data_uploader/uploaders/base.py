import json
import os
import warnings
from abc import ABC
from collections.abc import Generator
from functools import reduce
from typing import Any

import httpx
import tenacity
from pydantic import BaseModel

from rag_data_uploader.utils.base_mappings import BaseSerializeableMapping
from rag_data_uploader.utils.exceptions import (
    CredentialError,
    DocumentFormatError,
    EnforceMappingError,
    ParsingModelError,
    VectorStoreConnectionError,
)

warnings.filterwarnings("ignore")

MAX_RETRIES = os.getenv("MAX_RETRIES", 2)
NUM_THREADS = os.getenv("NUM_THREADS", 20)
MAX_DOCUMENTS = 10000


class RAGDocumentUploader(ABC):
    """Base class for uploading RAG documents to document store"""

    max_documents = MAX_DOCUMENTS

    def __init__(
        self, url: str, username: str, password: str, *, allow_missing_fields: bool = False
    ):
        """Creates a document uploader instance.

        Args:
            url:          URL for the document store instance.
            username:     Username for accessing document store.
            password:     Password for accessing document store.
        """
        self.url = url
        self.username = username
        self.password = password
        self.req_kwargs = {
            "auth": (self.username, self.password),
            "verify": False,
        }
        self.contact_store(verbose=True)
        self.allow_missing_fields = allow_missing_fields

        self.indices_to_replace: list[str] = []
        self.total_errors: list[Any] = []

    def contact_store(self, verbose=False):
        """Method which checks document store connection.

        Args:
            verbose:    Boolean which indicates if URL and response
                        should be printed.
        """
        # Tries to get contact with the document store
        response = httpx.head(self.url, **self.req_kwargs)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            raise VectorStoreConnectionError(response.text)
        if verbose:
            print("URL: ", self.url)
            print(
                f"Response from document store at URL {self.url}: ",
                response.text if response.text else "OK",
            )

    def upload_from_folder(
        self,
        folder: str,
        index: str,
        alias: str | None = None,
        mapping: BaseSerializeableMapping | None = None,
        parsing_model: BaseModel | None = None,
        batch_size: float = 1,
        threading: bool = True,
    ) -> None:
        """Method which uploads documents from folder.

        Reads documents from folder in mounted volume, preprocesses and uploads these to
        a document store using httpx library.

         Args:
             folder:        String which indicates the folder name in the mounted volume.
             index:         String which indicates the index name for the document store.
             alias:         String which gives the alias name which the index should use.
             mapping:       Index mapping, or None.
             parsing_model: A Pydantic model to use for preprocessing of documents.
                            The model should not be instantiated.
             batch_size:    Float which gives the maximum batch size, in Mb.
             threading:     Bool which indicates if threading should be used for uploading
                            documents or not.
        """
        try:
            assert parsing_model is not None
            assert issubclass(parsing_model, BaseModel)
        except (AssertionError, TypeError):
            msg = "Please provide an uninstantiated Pydantic parsing model to use."
            raise ParsingModelError(msg)

        # Check connection with vector store
        self.contact_store()

        self.index = index
        self.alias = alias
        self.mapping = mapping.model_dump() if mapping else None
        self.folder = folder

        self.indices_to_replace.clear()
        self.total_errors.clear()

        self._check_index_and_mapping()

        files = self._get_file_list(folder)
        total_documents = 0

        for batch in self._document_generator(files):
            documents_to_upload = self._preprocess_documents(batch, parsing_model=parsing_model)
            self._upload_documents(documents_to_upload, batch_size=batch_size, threading=threading)
            total_documents += len(documents_to_upload)

        print(
            "Number of successfully uploaded documents: ",
            total_documents - len(self.total_errors),
        )
        if self.total_errors:
            self._save_errors()
            print(f"Index {self.index} was not added to alias due to upload errors")
            self._remove_indices_from_alias(indices=[self.index])

        elif self.indices_to_replace:
            self._remove_indices_from_alias()

    def upload_documents(
        self,
        documents: list | dict,
        index: str,
        alias: str | None = None,
        mapping: BaseSerializeableMapping | None = None,
        parsing_model: BaseModel | None = None,
        batch_size: float = 1,
        threading: bool = True,
    ) -> None:
        """Method which uploads documents from memory.

        Takes JSON objects as input, preprocesses and uploads these to
        a document store using httpx library.

        Args:
            documents:      Either a list of JSON objects or a single JSON object
            index:          String which indicates the index name for the document store.
            alias:          String which gives the alias name which the index should use.
            mapping:        Index mapping, or None.
            parsing_model:  A Pydantic model to use for preprocessing of documents.
                            The model should not be instantiated.
            batch_size:     Float which gives the maximum batch size, in Mb.
            threading:      Bool which indicates if threading should be used for uploading
                            documents or not.
        """

        try:
            assert parsing_model is not None
            assert issubclass(parsing_model, BaseModel)
        except (AssertionError, TypeError):
            msg = "Please provide an uninstantiated Pydantic parsing model to use."
            raise ParsingModelError(msg)

        if isinstance(documents, dict):
            documents = [documents]
        elif not isinstance(documents, list) or (
            isinstance(documents, list) and not all(isinstance(doc, dict) for doc in documents)
        ):
            raise DocumentFormatError("Document(s) must be JSON object(s)")

        # Check connection with vector store
        self.contact_store()

        self.index = index
        self.alias = alias
        self.mapping = mapping.model_dump() if mapping else None

        self.indices_to_replace.clear()
        self.total_errors.clear()

        self._check_index_and_mapping()

        documents_to_upload = self._preprocess_documents(documents, parsing_model)

        print("Number of documents:", len(documents_to_upload))
        self._upload_documents(documents_to_upload, batch_size=batch_size, threading=threading)
        print(
            "Number of successfully uploaded documents: ",
            len(documents_to_upload) - len(self.total_errors),
        )
        if self.total_errors:
            self._save_errors()
            print(f"Index {self.index} was not added to alias due to upload errors")
            self._remove_indices_from_alias(indices=[self.index])

        elif self.indices_to_replace:
            self._remove_indices_from_alias()

    def delete_index(self, index: str | list) -> None:
        """Method which deletes an index.

        Checks that the user has the right to delete index by
        asking for password.

        Args:
            index:    The index name(s) to delete.
        """

        password = input("Provide password for document store: ")
        if password != self.password:
            raise CredentialError("Password is incorrect")
        else:
            self._delete_index(index)

    def _check_index_and_mapping(self) -> None:
        """Checks if an index exists.

        If the index does not exist, creates the index and changes the mapping
        to the alias mapping, if the alias exists.
        """

        # See if the current index exists
        response = httpx.head(f"{self.url}/{self.index}", **self.req_kwargs)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            self._create_index()
        # If index exists, get mapping
        else:
            self.mapping = self._get_mapping()

    def _document_generator(self, files: list) -> Generator[list[dict[str, Any]]]:
        """Method which generates a list of documents.

        The batch size is configured by MAX_DOCUMENTS.

        Args:
            files:      A list of file names to upload.

        Yields:
            documents:  A generator which prodices lists of
                        loaded documents to upload.
        """

        buffer = []
        documents = []
        for file in files:
            documents.extend(self._read_file(self.folder, file))
            if len(documents) > self.max_documents:
                buffer = documents[self.max_documents :]
                yield documents[: self.max_documents]
                documents = buffer
        yield documents

    def _read_file(self, folder: str, file: str) -> list[str]:
        """Helper function which reads a file and returns documents.

        Args:
            folder:         String which indicates the folder name
                            in the mounted volume.
            file:           String which gives the file name to be loaded

        Returns:
            documents:      A list of JSON objects
        """

        file_path = f"{folder}/{file}"
        try:
            with open(file_path, "rb") as f:
                return json.load(f)
        except Exception as e:
            print(file_path)
            raise e

    def _get_file_list(self, folder: str) -> list[str]:
        """Method which recursively gets file names from a potentially nested folder.

        Args:
            folder:   A folder name from which to get file names

        Returns:
            files:   A list of complete file paths

        """

        root, dirs, files = next(os.walk(folder))
        if dirs:
            for directory in dirs:
                if "zip" not in directory:
                    directory_files = self._get_file_list(f"{root}/{directory}")
                    print(f"Number of files in {directory}: {len(directory_files)}")
                    files.extend([f"{directory}/{file}" for file in directory_files])
        return [file for file in files if file.endswith(".json")]

    def _preprocess_documents(
        self, documents: list[dict[str, Any]], parsing_model: BaseModel
    ) -> list[dict[str, Any]]:
        """Preprocessing method.

        Method which takes in a list of JSON documents and modifies their content.
        This method also makes sure that no key-value pairs in documents that
        are not present in the current mapping are uploaded.

        Args:
            documents:   A list of dictionaries containing CPI documents

        Returns:
            documents:   A list of dictionaries with modified content.

        """

        documents = [parsing_model.model_validate(d).model_dump() for d in documents]

        # Match mapping to existing index, if it exists
        if self.mapping:
            # Remove document keys which are not present in mapping
            documents = [
                {k: v for k, v in d.items() if k in self.mapping["properties"].keys()}
                for d in documents
            ]

            if not self.allow_missing_fields:
                # Check if fields are missing from documents.
                # If any document is missing any fields, the upload will be cancelled.
                if difference := reduce(
                    lambda a, b: a.union(b),
                    (
                        set(self.mapping["properties"].keys()).difference(list(d.keys()))
                        for d in documents
                    ),
                ):
                    raise EnforceMappingError(
                        "All documents do not agree with mapping!",
                        "These fields are missing from at least some documents:",
                        difference,
                    )

        if not all(d.get("content", "") for d in documents):
            print(
                "Some documents had an empty or non-existing content field and were excluded from upload"
            )
        # Just keep documents which have a non-empty content field
        return [d for d in documents if d.get("content", "")]

    def _create_index(self) -> None:
        """Method which creates an index in the document store.

        Should be implemented for each subclass of DocumentUploader.
        """

        raise NotImplementedError

    @tenacity.retry(
        wait=tenacity.wait_exponential_jitter(),
        stop=tenacity.stop_after_attempt(MAX_RETRIES),
        reraise=True,
    )
    def _get_mapping(self):
        """Method which gets the mapping from an existing index.

        Should be implemented for each subclass of DocumentUploader.
        """

        raise NotImplementedError

    def _upload_documents(self, documents: list, batch_size: float, threading: bool = True):
        """Method which uploads documents to the document store.

        Should be implemented for each subclass of DocumentUploader.
        """

        raise NotImplementedError

    def _get_errors(self, response):
        """Helper method which extracts errors from the document upload response.

        Should be implemented for each subclass of DocumentUploader and called from _upload_documents.
        """

        raise NotImplementedError

    def _save_errors(self):
        """Method which saves errors which happen during the upload process."""

        print("There were errors in the upload process.")
        print("Errors have been saved to the current directory.\n")
        with open("upload_errors.json", "w") as f:
            json.dump(self.total_errors, f, indent=4)
        print("Example of an error: ")
        print(json.dumps(self.total_errors[0], indent=4))

    def _delete_index(self, index: str) -> None:
        """Method which deletes index from document store.

        Should be implemented for each subclass of DocumentUploader.
        """

        raise NotImplementedError

    def _find_alias_indices(self, alias: str | None = None):
        """Method which finds indices associated with alias.

        Finds the indices which are associated with the current alias
        and match the current index name.

        Args:
            alias:    Name of the alias to find indices for.
        """

        raise NotImplementedError

    def _remove_indices_from_alias(
        self, alias: str | None = None, indices: str | list[str] | None = None
    ):
        """Method which removes indices from the current alias.

        Args:
            alias:    Name of the alias to remove indices from.
            indices:  List of indices to remove.
        """

        raise NotImplementedError
