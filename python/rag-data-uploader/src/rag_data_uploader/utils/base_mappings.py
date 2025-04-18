from abc import abstractmethod
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_serializer,
    model_validator,
)


class TypeMappingES(BaseModel):
    type: str

    model_config = ConfigDict(validate_default=True)

    @field_validator("type", mode="after")
    @classmethod
    def validate_type(cls, v) -> str:
        assert v in (
            allowed := ("long", "float", "keyword", "boolean", "text")
        ), f"Allowed types: {allowed}"
        return v


class EmbeddingMappingES(TypeMappingES):
    type: str
    dims: int

    model_config = ConfigDict(validate_default=True)

    @field_validator("type", mode="after")
    @classmethod
    def validate_type(cls, v) -> str:
        assert v in (
            allowed := ("dense_vector", "knn_vector")
        ), f"Allowed embedding type(s): {allowed}"
        return v


class EmbeddingMappingOS(TypeMappingES):
    type: str
    dimension: int

    model_config = ConfigDict(validate_default=True)

    @field_validator("type", mode="after")
    @classmethod
    def validate_type(cls, v) -> str:
        assert v in (
            allowed := ("dense_vector", "knn_vector")
        ), f"Allowed embedding type(s): {allowed}"
        return v


class BaseSerializeableMapping(BaseModel):
    @model_serializer
    def serialize_model(self) -> dict[str, Any]:
        field_mappings = {k: v for k, v in self if v}
        mapping = {"properties": field_mappings}
        return mapping

    def __str__(self) -> str:
        return self.model_dump_json()

    def __repr__(self) -> str:
        return self.model_dump_json()


class BaseObjectMapping(BaseSerializeableMapping):
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def validate_fields(cls, data: Any) -> dict:
        tmp_data = {}
        for k, v in data.items():
            if isinstance(v, str):
                tmp_data[k] = TypeMappingES(type=v)
            elif isinstance(v, dict):
                tmp_data[k] = BaseObjectMapping(**v)
        return tmp_data


class TemplateMappingES(BaseModel):
    name: str
    match: str
    to: str

    @model_serializer
    def serialize_template(self) -> list[dict[Any, Any]]:
        return [
            {
                self.name: {
                    "match_mapping_type": self.match,
                    "mapping": TypeMappingES(type=self.to),
                }
            }
        ]


class ESBaseMapping(BaseSerializeableMapping):
    model_config = ConfigDict(validate_default=True, extra="allow")

    @field_validator("*", mode="after")
    @classmethod
    def validate_fields(cls, v, info) -> None | TypeMappingES:
        if info.field_name == "embedding":
            return v if not v else cls.embedding_mapping(dims=v)
        elif info.field_name == "template" and v is not None:
            return TemplateMappingES(**v)
        elif isinstance(v, dict):
            return BaseObjectMapping(**v)
        elif v is not None:
            return TypeMappingES(type=v)
        else:
            return None

    @model_serializer
    def serialize_model(self) -> dict[str, Any]:
        field_mappings = {k: v for k, v in self if v and k != "template"}
        mapping = {"properties": field_mappings}
        if "template" in self.model_fields:
            mapping.update({"dynamic_templates": self.template})
        return mapping

    @classmethod
    @abstractmethod
    def embedding_mapping(cls, **kwargs) -> TypeMappingES:
        raise NotImplementedError


class ES7BaseMapping(ESBaseMapping):
    content: str = "text"
    content_type: str = "keyword"
    embedding: int | None = None

    @classmethod
    def embedding_mapping(cls, dims) -> TypeMappingES:
        return EmbeddingMappingES(type="dense_vector", dims=dims)


class OSBaseMapping(ESBaseMapping):
    content: str = "text"
    content_type: str = "keyword"
    embedding: int | None = None

    @classmethod
    def embedding_mapping(cls, dims) -> TypeMappingES:
        return EmbeddingMappingOS(type="knn_vector", dimension=dims)
