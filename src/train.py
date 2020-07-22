#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple training script
"""

from __future__ import annotations

import logging
import os
import pickle

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
from src.load_word_embedding import load_word_embedding
from src.logger import logger
from src.utils import read_jsonl
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from src.classifier import CNN

app = typer.Typer()


np.random.seed(1337)

# Note that type hints seem to fail in some cases due to issue with typer
# So I've not included them here.
@app.command()
def train(
    test_data_path="data/processed/test.jsonl",
    train_data_path="data/processed/train.jsonl",
    output_path="models",
    model_output_path="models",
    embedding_path="data/raw/glove.6B.50d.txt",
    embedding_dim=50,
    batch_size=1024,
    epochs=3,
    learning_rate=0.01,
    lowercase=True,
    seq_length=1000,
    num_words=1000,
    oov_token="<OOV>",
    padding_style="pre",
    trunc_style="pre",
    checkpoint=True,
    checkpoint_path="models"
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

    cnn.load_data(train_data_path, test_data_path)

    # Prepare the data with tokenisation, padding, etc.

    cnn.prep_data(
        oov_token=oov_token,
        trunc_style=trunc_style,
        padding_style=padding_style,
        num_words=int(num_words),
        lowercase=lowercase,
        save_tokenizer=True,
    )

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
