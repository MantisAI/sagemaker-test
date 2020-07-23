#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple training script
"""

from __future__ import annotations

import logging
import os

import numpy as np
import tensorflow as tf
import typer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.CNN import CNN
from src.logger import logger

app = typer.Typer()


np.random.seed(1337)

# Note that type hints seem to fail in some cases due to issue with typer
# So I've not included them here.
@app.command()
def train(
    processed_path="data/processed",
    output_path="models",
    model_output_path="models",
    embedding_path="data/raw/glove.6B.50d.txt",
    embedding_dim=50,
    batch_size=1024,
    epochs=2,
    learning_rate=0.01,
    seq_length=1000,
    checkpoint=True,
    checkpoint_path="models",
):
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

    cnn.load_train_test_data(processed_path)

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
