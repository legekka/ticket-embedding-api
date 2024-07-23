FROM ubuntu:22.04

ARG PYTHON_VERSION=3.11
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev python3-pip python${PYTHON_VERSION} && \
    rm -rf /var/lib/apt/lists/*

RUN adduser --no-create-home --home /opt/aiops aiops
RUN mkdir -p /opt/aiops; chown aiops:aiops /opt/aiops

WORKDIR /opt/aiops
USER aiops

COPY . /opt/aiops

ARG DB_PATH="/opt/aiops/database"
ENV DB_PATH="${DB_PATH}"
ARG DB_CONFIG="/opt/aiops/database/config.json"
ENV DB_CONFIG="${DB_CONFIG}"

ENV PATH=${PATH}:/opt/aiops/.local/bin
RUN pip install -r requirements.txt

EXPOSE 8000
CMD fastapi run api.py