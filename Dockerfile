FROM python:3.7-slim-buster as base

FROM base as builder

RUN apt-get update && apt-get install -y \
    python3-dev \
    gcc \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip3 install -r requirements.txt -t /install --upgrade
RUN pip3 install sagemaker-training -t /install --upgrade

FROM base as app

COPY --from=builder /install/ /usr/local/lib/python3.7/site-packages/

WORKDIR /opt/ml/
COPY src/mnist_cnn.py ./code/train.py

# define train.py as the script entry point
ENV SAGEMAKER_PROGRAM train.py

