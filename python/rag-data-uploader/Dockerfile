ARG BASE_IMAGE="python:3.13.5-slim-bullseye"

FROM ${BASE_IMAGE}

LABEL maintainer="David Rommedahl <david.rommedahl@ericsson.com>"
LABEL title="RAG data uploader module"
LABEL description="Implements classes to help with uploading RAG documents to vector stores"

WORKDIR /app

COPY rag_data_uploader /app/rag_data_uploader
COPY pyproject.toml /app/pyproject.toml
COPY VERSION /app/VERSION

RUN pip install .