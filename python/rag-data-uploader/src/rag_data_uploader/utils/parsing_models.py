"""Basic Pydantic parsing models for the RegenX data sources."""

import datetime
from typing import Literal

from pydantic import (
    AliasChoices,
    AliasPath,
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)


def metadata_field(field_name: str, **kwargs) -> Field:
    """Helper function which constructs paths for a given metadata field.

    This helper function returns a pydantic.Field object, and it takes any
    arguments that the Field argument takes. If 'validation_alias' is specified
    it is dropped.
    """
    kwargs.pop("validation_alias", None)
    return Field(
        validation_alias=AliasChoices(
            field_name,
            AliasPath("metadata", field_name),
            AliasPath("metadata", "builder_meta", field_name),
        ),
        **kwargs,
    )


class RegenXBaseParsingModel(BaseModel):
    """Base parsing model with required fields.

    Any parsing model should inherit from this base model. Desired metadata fields,
    field validators and serializers should be added to the child class.
    """

    model_config = ConfigDict(extra="ignore")

    content: str = Field(validation_alias=AliasChoices("page_content", "content"))
    content_type: Literal["text"] = "text"
    embedding: list[float]


class RegenXParsingModelCPI(RegenXBaseParsingModel):
    """Parsing model for RegenX CPI data.

    This model inherits from the base parsing model and extends it
    with required metadata fields for the RegenX CPI schema.
    """

    class ContentFromLibraries(BaseModel):
        """Sub-model to be used for nested dictionary within CPI document."""

        model_config = ConfigDict(extra="ignore")

        identity: str
        title: str
        date: datetime.date

        @field_serializer("date")
        def serialize_date(self, v: datetime.date) -> str:
            return str(v)

    source: str = metadata_field("source")
    document_number: str = metadata_field("document_number")
    identity: str = metadata_field("identity")
    chapter_title: str | list[str] = metadata_field("chapter_title")
    revision: str = metadata_field("revision")
    title: str = metadata_field("title")
    document_type: str = metadata_field("document_type")
    category_tree: str = metadata_field("category_tree")
    document_url: AnyHttpUrl = metadata_field("document_url")
    library_url: AnyHttpUrl = metadata_field("library_url")
    external_document_url: AnyHttpUrl = metadata_field("external_document_url")
    content_from_libraries: list[ContentFromLibraries] = metadata_field("content_from_libraries")
    eridoc_document_number: str = metadata_field("eridoc_document_number")

    @field_validator("chapter_title")
    @classmethod
    def validate_chapter_title(cls, v: str | list[str]) -> str:
        """Ensures that chapter_title is a string instead of a list of strings."""
        if isinstance(v, list):
            if len(v) == 1:
                return next(iter(v))
            else:
                raise ValueError
        return v

    @field_serializer("document_url", "library_url", "external_document_url")
    def serialize_url(self, v: AnyHttpUrl) -> str:
        """URL type is not JSON serializable."""
        return str(v)
