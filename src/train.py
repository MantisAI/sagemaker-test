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
from src.CNN import CNN
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
    batch_size=params["train"]["batch-size"],
    checkpoint=params["train"]["checkpoint"],
    checkpoint_path=params["train"]["checkpoint-path"],
    embedding_dim=params["train"]["embedding-dim"],
    embedding_path=params["train"]["embedding-path"],
    epochs=params["train"]["epochs"],
    learning_rate=params["train"]["learning-rate"],
    model_output_path=params["train"]["model-output-path"],
    output_path=params["train"]["output-path"],
    processed_folder=params["common"]["processed-folder"],
    seq_length=params["train"]["seq-length"],
):
    # Capture parameters that have been used in the dvc pipeline, but not
    # directly here.

    with start_run():

        log_param(
            "train_s3_file",
            dvc.api.get_url(
                os.path.join(params["common"]["processed-folder"], "train.npz")
            ),
        )
        log_param(
            "test_s3_file",
            dvc.api.get_url(
                os.path.join(params["common"]["processed-folder"], "test.npz")
            ),
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

        cnn = CNN(
            output_path=output_path,
            model_output_path=model_output_path,
            seq_length=int(seq_length),
        )

        # Load the data from disk

        cnn.load_train_test_data(processed_folder)

        cnn.load_indices()

        # Load word embedding from disk

        cnn.load_word_embedding(
            embedding_path=embedding_path, embedding_dim=int(embedding_dim),
        )

        # Load callbacks. These are consumed in the fit method

        cnn.load_callbacks(checkpoint=checkpoint, checkpoint_path=checkpoint_path)

        # Fit the model

        cnn.fit(
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
        )


if __name__ == "__main__":
    app()
