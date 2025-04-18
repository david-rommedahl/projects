import warnings

import httpx

from rag_data_uploader.uploaders.elasticsearch import ElasticsearchUploader
from rag_data_uploader.utils.exceptions import IndexCreateError, MappingNotProvidedError

warnings.filterwarnings("ignore")
# 512 is the default value in Langchain
KNN_EF_SEARCH = 512


class OpensearchUploader(ElasticsearchUploader):
    """Class which uploads documents to an Opensearch instance.

    This API is identical to Elasticsearch, with the exception
    of the _create_index method where it has to be specified if the index should be used for kNN search.
    """

    def _create_index(self) -> None:
        """Method which creates an index in Opensearch.

        In Opensearch, an index has to be created with a
        knn value set to True to perform kNN search.
        """
        if not self.mapping:
            raise MappingNotProvidedError("Mapping must be specified for non-existing indices")

        body = {"mappings": self.mapping}
        # Check if an embedding field is set in mapping. If so, update the body used for creating the index
        # to allow for kNN search
        if "embedding" in self.mapping["properties"].keys():
            body.update(
                {
                    "settings": {
                        "index": {
                            "knn": True,
                            "knn.algo_param.ef_search": KNN_EF_SEARCH,
                        }
                    }
                }
            )
        if self.alias:
            self._find_alias_indices()
            body.update({"aliases": {self.alias: {}}})
            self._enforce_alias_mapping()

        try:
            _ = httpx.put(
                f"{self.url}/{self.index}",
                json=body,
                **self.req_kwargs,
            ).raise_for_status()
        except httpx.HTTPStatusError as err:
            raise IndexCreateError(
                f"Something went wrong with creating the index: {err.response.text}"
            )
