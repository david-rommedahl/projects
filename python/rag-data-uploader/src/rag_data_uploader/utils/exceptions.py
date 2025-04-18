class PayloadTooLargeError(Exception):
    """Raised when a post request is rejected because payload is too large"""

    pass


class MappingNotProvidedError(Exception):
    """Raised when the user tries to upload documents to an index which does not exist, without specifying a mapping"""

    pass


class IndexCreateError(Exception):
    """Raised when something goes wrong with creating an index"""

    pass


class BulkPostError(Exception):
    """Raised when an error is encountered while trying to upload documents using bulk API"""

    pass


class GetMappingError(Exception):
    """Raised when an error is thrown while trying to get mapping for an existing index"""

    pass


class EnforceMappingError(Exception):
    """Raised when a user-defined mapping is in conflict with the alias mapping"""

    pass


class DocumentFormatError(Exception):
    """Raised if the provided documents are not either a JSON object or a list of JSON objects"""

    pass


class CredentialError(Exception):
    """Raised when the wrong password is provided"""

    pass


class VectorStoreConnectionError(Exception):
    """Raised if there is a problem with the connection to vector store."""

    pass


class ParsingModelError(Exception):
    """Raised if no parsing model is provided for upload."""

    pass
