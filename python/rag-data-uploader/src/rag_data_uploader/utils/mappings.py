from pydantic import ConfigDict

from rag_data_uploader.utils.base_mappings import ES7BaseMapping, OSBaseMapping


class ES7MappingCPI(ES7BaseMapping):
    source: str = "keyword"
    seq_num: str = "keyword"
    book_id: str = "keyword"
    book_number: str = "keyword"
    book_pdf: str = "keyword"
    book_edition: str = "keyword"
    book_title: str = "keyword"
    book_alt_title: str = "keyword"
    book_view_permission: str = "keyword"
    book_audience: str = "keyword"
    book_date: str = "keyword"
    doctype: str = "keyword"
    date: str = "keyword"
    view_permission: str = "keyword"
    products: str = "keyword"
    embedding: int = 1536
    cpi_folders: str = "keyword"
    pia_graphs: str = "keyword"


class ES7MappingNewCPI(ES7BaseMapping):
    book_alt_title: str = "keyword"
    book_id: str = "keyword"
    book_number: str = "keyword"
    book_pdf: str = "keyword"
    book_title: str = "keyword"
    categories: str = "keyword"
    chapter_title: str = "keyword"
    content_type: str = "keyword"
    products: str = "keyword"
    pia_graphs: str = "keyword"
    title: str = "keyword"


class ES7Mapping3GPP(ES7BaseMapping):
    content_type: str = "keyword"
    embedding: int = 768
    source: str = "keyword"
    title: str = "keyword"
    short_title: str = "keyword"
    full_title: str = "keyword"
    release: str = "keyword"
    section_id: str = "keyword"
    spec_id: str = "keyword"
    sections: dict[str, str] = {"chapter": "keyword"}
    template: dict[str, str] = {
        "name": "template_3gpp",
        "match": "string",
        "to": "keyword",
    }


class OSMapping3GPP(OSBaseMapping):
    content_type: str = "keyword"
    embedding: int = 768
    source: str = "keyword"
    title: str = "keyword"
    short_title: str = "keyword"
    full_title: str = "keyword"
    release: str = "keyword"
    section_id: str = "keyword"
    spec_id: str = "keyword"
    sections: dict[str, str] = {"chapter": "keyword"}
    template: dict[str, str] = {
        "name": "template_3gpp",
        "match": "string",
        "to": "keyword",
    }


class OSRegenXMappingCPI(OSBaseMapping):
    model_config = ConfigDict(validate_default=True, extra="ignore")

    embedding: int = 1536
    identity: str = "keyword"
    source: str = "keyword"
    document_number: str = "keyword"
    chapter_title: str = "keyword"
    revision: str = "keyword"
    title: str = "keyword"
    document_type: str = "keyword"
    category_tree: str = "keyword"
    document_url: str = "keyword"
    library_url: str = "keyword"
    external_document_url: str = "keyword"
    content_from_libraries: dict[str, str] = {
        "identity": "keyword",
        "title": "keyword",
        "date": "keyword",
    }
    eridoc_document_number: str = "keyword"
