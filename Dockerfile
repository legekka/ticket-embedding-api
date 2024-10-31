ARG IMAGE=pytorch/pytorch
ARG TAG=2.4.1-cuda12.4-cudnn9-runtime

FROM ${IMAGE}:${TAG} AS base

RUN apt-get update \
    && apt-get -y install libpq-dev gcc

RUN adduser --no-create-home --home /opt/aiops aiops
RUN mkdir -p /opt/aiops; chown aiops:aiops /opt/aiops

WORKDIR /opt/aiops
USER aiops

COPY . /opt/aiops

# Setting up the environment variables
ARG DB_PATH="/opt/aiops/database"
ENV DB_PATH="${DB_PATH}"
ARG DB_CONFIG="/opt/aiops/database/config.json"
ENV DB_CONFIG="${DB_CONFIG}"
ARG HFMODEL="NYTK/PULI-BERT-Large"
ENV HFMODEL="${HFMODEL}"

ARG IRISDB_NAME="irisdb"
ENV IRISDB_NAME="${IRISDB_NAME}"
ARG IRISDB_USER="irisuser"
ENV IRISDB_USER="${IRISDB_USER}"
ARG IRISDB_PASSWORD="irispwd"
ENV IRISDB_PASSWORD="${IRISDB_PASSWORD}"
ARG IRISDB_HOST="localhost"
ENV IRISDB_HOST="${IRISDB_HOST}"
ARG IRISDB_PORT="5432"
ENV IRISDB_PORT="${IRISDB_PORT}"

ENV PATH=${PATH}:/opt/aiops/.local/bin
RUN pip install -r requirements.txt

EXPOSE 8000
CMD fastapi run api.py