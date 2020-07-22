import logging
import os

import dvc.api
from sagemaker.estimator import Estimator

HOST = os.getenv("HOST")
REPO = os.getenv("REPO")
VERSION = os.getenv("VERSION")

ROLE_ARN = os.getenv("ROLE_ARN")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE")


train = dvc.api.get_url("data/processed/train.jsonl")
test = dvc.api.get_url("data/processed/test.jsonl")
word_embedding = dvc.api.get_url("data/raw/glove.6B.50d.txt")

train_file = os.path.split(train)[-1]
test_file = os.path.split(test)[-1]
word_embedding_file = os.path.split(word_embedding)[-1]

input_channels = {
    "train": train,
    "test": test,
    "word_embedding": word_embedding,
    # Setting these to file:// will upload the data from the local drive
    # "train": "file://data/processed/train.jsonl",
    # "test": "file://data/processed/test.jsonl",
    # "word_embedding": "file://data/raw/glove.6B.50d.txt",
}

estimator = Estimator(
    image_name=f"{HOST}/{REPO}:{VERSION}",
    role=ROLE_ARN,
    train_instance_count=1,
    train_instance_type=INSTANCE_TYPE,
    hyperparameters={
        "test-data-path": "/opt/ml/input/data/test/" + test_file,
        "train-data-path": "/opt/ml/input/data/train/" + train_file,
        "output-path": "/opt/ml/output/",
        "model-output-path": "/opt/ml/model/",
        "embedding-path": "/opt/ml/input/data/word_embedding/" + word_embedding_file,
        "embedding-dim": 50,
        "batch-size": 1024,
        "epochs": 20,
        "learning-rate": 0.01,
        "lowercase": True,
        "num-words": 1000,
        "seq-length": 1000,
        "oov-token": "<OOV>",
        "padding-style": "pre",
        "trunc-style": "pre",
    },
)

estimator.fit(inputs=input_channels)
