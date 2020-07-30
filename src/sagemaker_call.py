#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os

import dvc.api
import typer
import yaml
from sagemaker.estimator import Estimator

app = typer.Typer()

REPO_URL = os.getenv("REPO_URL")
VERSION = os.getenv("VERSION")
ROLE_ARN = os.getenv("ROLE_ARN")

train = dvc.api.get_url("data/processed/train.npz")
test = dvc.api.get_url("data/processed/test.npz")
word_embedding = dvc.api.get_url("data/raw/glove.6B.50d.txt")
indices = dvc.api.get_url("models/indices.pickle")

train_file = os.path.split(train)[-1]
test_file = os.path.split(test)[-1]
word_embedding_file = os.path.split(word_embedding)[-1]
indices_file = os.path.split(indices)[-1]


@app.command()
def main(
    gpu: bool = typer.Option(
        False, "--gpu",
        help="Should a GPU based docker image be used? If this flag is set, and you are running a SageMaker job, you must specify an instance with a GPU (e.g. ml.p2/3...).",
    ),
    instance_type: str = typer.Option(
        "local",
        help="SageMaker instance used to run the model, e.g. ml.p2.xlarge or ml.c5.xlarge. Setting this to local will run the container locally.",
    ),
):

    image_name = f"{REPO_URL}:{VERSION}"

    if gpu:
        image_name = image_name + "-gpu"

    input_channels = {
        "train": train,
        "test": test,
        "word_embedding": word_embedding,
        "indices": indices,
        # Setting these to file:// will upload the data from the local drive
        # "train": "file://data/processed/train.jsonl",
        # "test": "file://data/processed/test.jsonl",
        # "word_embedding": "file://data/raw/glove.6B.50d.txt",
    }
    estimator = Estimator(
        image_name=image_name,
        role=ROLE_ARN,
        train_instance_count=1,
        train_instance_type=instance_type,
        hyperparameters={
            "test-path": "/opt/ml/input/data/test/" + test_file,
            "train-path": "/opt/ml/input/data/train/" + train_file,
            "indices-path": "/opt/ml/input/data/indices/" + indices_file,
            "output-path": "/opt/ml/model/",
            "model-output-path": "/opt/ml/model/",
            "embedding-path": "/opt/ml/input/data/word_embedding/"
            + word_embedding_file,
            "embedding-dim": 50,
            "batch-size": 1024,
            "epochs": 2,
            "learning-rate": 0.01,
            "seq-length": 1000,
            "checkpoint": True,
            "checkpoint-path": "/opt/ml/model/",
        },
    )

    estimator.fit(inputs=input_channels)


if __name__ == "__main__":
    app()
