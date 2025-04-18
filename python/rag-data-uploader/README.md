# rag-data-uploader

**NOTE: This package is a work in progress and is being converted from an earlier, more specific, stage to a new general stage.**

## Description
This package implements classes for uploading RAG documents to a data store.  These uploader classes uploads JSON objects to either a vector store instance. The documents can either be uploaded from files or from JSON objects in memory of a running program. 

This package also implements default mappings for a few document types and store types. One default mapping for CPI documents with an embedding field for Elasticsearch7, one for CPI documents without embedding, as well as one for 3GPP documents for Elasticsearch7 and one for Opensearch or Elasticsearch8. The user can supply their own custom mappings for each upload.

There are also Pydantic parsing models to be used for data pre-processing, which can be passed in to the different uploading methods of the uploaders.

## Usage

### Data Uploaders
The uploader classes can be used as they are, or sub-classed to create custom classes for new vector stores or data sources.

The default method for uploading documents from JSON files is
`RAGDataUploader.upload_from_folder()`:

```python
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
```
This method uploads documents from a folder on the machine. It is done in batches, to avoid having to read all data into memory at once, and using threading to speed up the process. 

Another method is `RAGDataUploader.upload_documents()`, which uploads JSON objects already in memory to the vector store using a similar procedure.

```python
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
```

See the [uploader demo notebook](https://github.com/david-rommedahl/projects/blob/main/python/rag-data-uploader/notebooks/uploader_demo.ipynb) for an example of how to upload documents to a document store running on `https://localhost:9200`.

### Mappings

This module implements Pydantic models as mappings to simplify using, extending and creating your own mappings which are consistent with the datastore that you are using. Currently, mapping classes are implemented for Elasticsearch 7 and Opensearch, since those are the two vector stores that are currently in use in PIA. To use a default mapping class, all the user has to do is import it from the mapping module and pass an instance of that imported mapping class to one of the upload methods of the desired data uploader class. To use the mapping class `ES7BaseMapping`, the user would do this:

```python
from rag_data_uploader.utils.mappings import ES7BaseMapping

mapping = ES7BaseMapping()
```

For a more in-depth demo of how to work with the different mapping classes, see the [mapping demo notebook](https://github.com/david-rommedahl/projects/blob/main/python/rag-data-uploader/notebooks/mapping_demo.ipynb).

### Parsing Models

Data preprocessing is done using Pydantic parsing models, and these should be passed in to the uploading method used for uploading documents to the vector store. This makes the uploading process more flexible, as the parsing model used can be switched out depending on the data source. See the [parsing model notebook](https://github.com/david-rommedahl/projects/blob/main/python/rag-data-uploader/notebooks/parsing_model_test.ipynb) for a more in-depth demonstration of how to use these parsing models.








