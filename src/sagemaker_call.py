import os

from sagemaker.estimator import Estimator

HOST = os.getenv("HOST")
REPO = os.getenv("REPO")
VERSION = os.getenv("VERSION")

ROLE_ARN = os.getenv("ROLE_ARN")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE")

estimator = Estimator(
    image_name=f"{HOST}/{REPO}:{VERSION}-gpu",
    role=ROLE_ARN,
    train_instance_count=1,
    train_instance_type=INSTANCE_TYPE,
    hyperparameters={
        "test-data-path": "/opt/ml/input/data/test/test.jsonl",
        "train-data-path": "/opt/ml/input/data/train/train.jsonl",
        "output-path": "/opt/ml/output/",
        "model-output-path": "/opt/ml/model/",
        "embedding-path": "/opt/ml/input/data/word_embedding/glove.6B.50d.txt",
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

estimator.fit(
    inputs={
        "train": "s3://muanalytics-data/sagemaker-test/data/processed/train.jsonl",
        "test": "s3://muanalytics-data/sagemaker-test/data/processed/test.jsonl",
        "word_embedding": "s3://muanalytics-data/sagemaker-test/data/raw/glove.6B.50d.txt",
        #"train": "file://data/processed/train.jsonl",
        #"test": "file://data/processed/test.jsonl",
        #"word_embedding": "file://data/raw/glove.6B.50d.txt",
    }
)
