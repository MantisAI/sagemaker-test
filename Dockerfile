FROM python:3.7.8-slim-buster as base

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    wget \
    zlib1g-dev \
    libreadline-gplv2-dev \
    libncursesw5-dev \
    libssl-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libc6-dev \
    libbz2-dev \
    tk-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir sagemaker-training

COPY src/train.py /opt/ml/code/train.py
COPY requirements.txt /opt/ml/code/requirements.txt

ENV SAGEMAKER_PROGRAM train.py
