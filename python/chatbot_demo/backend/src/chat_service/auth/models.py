from pydantic import BaseModel


class User(BaseModel):
    """The authenticated user for a request.

    A deliberately minimal identity carrying just the ``id`` used to scope
    conversations per user (the case's privacy requirement: users must not see
    each other's conversations). When real auth slots in, this model grows to
    carry the verified claims instead of a trusted header value.
    """

    id: str
