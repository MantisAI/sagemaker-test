ARG BASE

FROM $BASE as base

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
    git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

COPY . .

#
# NOTE: that by installing out requirements.txt we will potentially be overwriting
# dependencies set in the GPU_BASE image. This may cause a job to fail or the CPU
# to be used instead of the GPU and will likely result in a larger image. 
# The other approach is to just install manually here the dependencies which are
# not already installed on the base image. This may cause your local virtualenv 
# to be out of sync with the cotnainer which is undesireable.
#

RUN pip install --no-cache-dir -r requirements.txt \
        && pip install --no-cache-dir . \
        && pip install --no-cache-dir sagemaker-training \
        && rm requirements.txt \
        && cp ./src/train.py ./train.py

ARG MLFLOW_URI
ENV MLFLOW_URI=$MLFLOW_URI
ENV SAGEMAKER_PROGRAM train.py
