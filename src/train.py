#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple training script
"""

import logging
import os

import numpy as np
import yaml

import dvc.api
import mlflow.tensorflow
import tensorflow as tf
import typer
from mlflow import log_param, start_run
from src.model import Model
from src.logger import logger
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

app = typer.Typer()

mlflow.tensorflow.autolog()
mlflow.set_tracking_uri(os.getenv(("MLFLOW_URI")))

np.random.seed(1337)

params = yaml.safe_load(open("params.yaml"))


@app.command()
def train(
    train_path=params["common"]["train-path"],
    test_path=params["common"]["test-path"],
    indices_path=params["common"]["indices-path"],
    batch_size=params["train"]["batch-size"],
    checkpoint=params["train"]["checkpoint"],
    checkpoint_path=params["train"]["checkpoint-path"],
    embedding_dim=params["train"]["embedding-dim"],
    embedding_path=params["train"]["embedding-path"],
    epochs=params["train"]["epochs"],
    learning_rate=params["train"]["learning-rate"],
    model_output_path=params["train"]["model-output-path"],
    output_path=params["train"]["output-path"],
    seq_length=params["train"]["seq-length"],
):
    # Capture parameters here for MLFLow that are used in the dvc pipeline.

    with start_run():

        log_param(
            "train_data", dvc.api.get_url(params["common"]["train-path"]),
        )
        log_param(
            "test_data", dvc.api.get_url(params["common"]["test-path"]),
        )
        log_param(
            "indices", dvc.api.get_url(params["common"]["indices-path"]),
        )
        log_param(
            "word_embedding", params["train"]["embedding-path"],
        )

        log_param("embedding_dim", params["train"]["embedding-dim"])
        log_param("embedding_path", params["train"]["embedding-path"])
        log_param("lowercase", params["train"]["lowercase"])
        log_param("num_words", params["train"]["num-words"])
        log_param("oov_token", params["train"]["oov-token"])
        log_param("padding_style", params["train"]["padding-style"])
        log_param("trunc_style", params["train"]["trunc-style"])
        log_param("seq_length", params["train"]["seq-length"])
        log_param("test_prop", params["prepare"]["test-prop"])

        # Create output path if not exists

        for path in [output_path, model_output_path, checkpoint_path]:
            os.makedirs(path, exist_ok=True)

        # Set up logging

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh = logging.FileHandler(os.path.join(output_path, "log.txt"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Instantiate the model class

        model = Model(
            output_path=output_path,
            model_output_path=model_output_path,
            seq_length=int(seq_length),
        )

        # Load the data from disk

        model.load_train_test_data(test_path, train_path)

        model.load_indices(indices_path)

        # Load word embedding from disk

        model.load_word_embedding(
            embedding_path=embedding_path, embedding_dim=int(embedding_dim),
        )

        # Load callbacks. These are consumed in the fit method

        model.load_callbacks(checkpoint=checkpoint, checkpoint_path=checkpoint_path)

        # Fit the model

        model.fit(
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
        )


if __name__ == "__main__":
    app()
