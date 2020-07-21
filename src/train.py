#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple training script
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import List

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

app = typer.Typer()


np.random.seed(1337)


class TextClassifier:
    def __init__(
        self,
        output_path: str,
        model_output_path: str,
        seq_length: int = 1000,
        char_embedding: bool = True,
    ):
        """A binary classification model based on a simple CNN

        Args:
            output_path: Path to dir where model results will be saved. Will be
                created if not exists.
            seq_length: Length of tokens to be fed to the model as training
                examples. Anything longer than this will be truncated. Anything
                shorter will be padded.
            char_embedding: Should a character embedding be built.
        """

        self.output_path = output_path
        self.model_output_path = model_output_path
        self.seq_length = seq_length
        self.char_embedding = char_embedding

        # Defined in load_word_embedding

        self.embedding_dim = 0  # type: int
        self.history = None  # type: tf.keras.model.fit

    def load_data(self, test_data_path: str, train_data_path: str):
        """Load data from jsonl file

        Args:
            data_path: Path to jsonl file.
        """

        test_data = read_jsonl(test_data_path)
        train_data = read_jsonl(train_data_path)

        X_train, X_test, y_train, y_test = [], [], [], []

        for review in train_data:
            X_train.append(review["text"])
            y_train.append(review["label"])

        for review in test_data:
            X_test.append(review["text"])
            y_test.append(review["label"])

        self.y_test = np.array(y_test).astype(int)
        self.y_train = np.array(y_train).astype(int)
        self.X_test = np.array(X_test)
        self.X_train = np.array(X_train)

        logger.info("Train data shape: %s, %s", len(self.X_train), len(self.y_train))
        logger.info("Test data shape: %s, %s", len(self.X_test), len(self.y_test))

    def prep_data(
        self,
        oov_token: str = "<OOV>",
        trunc_style: str = "pre",
        padding_style: str = "pre",
        num_words: int = 10000,
        lowercase: bool = True,
        save_tokenizer: bool = False,
    ):
        """Prepare data for model fitting

        Args:
            oov_token: Token used for out of vocab tokens.
            trunc_style: One of ["pre", "post"]. Determines how truncating
                padding will be done: before or after the vector.
            padding_style: One of ["pre", "post"]. Determines how padding will
                be done: before or after the vector.
            num_words: Maximum number of tokens allowed in a training example.
                Anything shorter or longer will be padded or truncated using
                the respective style defined by trinc_style and padding_style.
            lowercase: Convert tokens to lowercase when tokenizing.
            save_tokenizer: Save tokenizer to pickle using `save_tokenizer()`.
        """

        self.tokenizer = Tokenizer(
            oov_token=oov_token, num_words=num_words, lower=lowercase
        )
        self.tokenizer.fit_on_texts(self.X_train)

        self.word_index = self.tokenizer.word_index

        if save_tokenizer:
            self.save_tokenizer()

        self.vocab_size = len(self.word_index)

        logger.info(f"Number of words in index: {self.vocab_size}")
        logger.info(f"Using {num_words} words")
        logger.info("Creating train sequences")

        # Training data

        X_train_sequences = self.tokenizer.texts_to_sequences(self.X_train)
        self.X_train_padded = pad_sequences(
            X_train_sequences,
            maxlen=self.seq_length,
            truncating=trunc_style,
            padding=padding_style,
        )

        logger.info("Creating test sequences")
        # Testing data

        X_test_sequences = self.tokenizer.texts_to_sequences(self.X_test)
        self.X_test_padded = pad_sequences(
            X_test_sequences,
            maxlen=self.seq_length,
            truncating=trunc_style,
            padding=padding_style,
        )

    def save_tokenizer(self):
        """Save tokenizer to pickle

        Always saves to a default location: output_path + tokenizer.pickle
        """
        with open(os.path.join(self.output_path, "tokenizer.pickle"), "wb") as f:
            pickle.dump(self.tokenizer, f)

    def load_word_embedding(self, embedding_path, embedding_dim):
        """Load and prepare word embedding

        Load word embedding, returning a dict indexed by token, with the
        embedding vector as values.

        Args:
            embedding_path: Path to word_embedding (a tab separated .txt file)
            embedding_dim: Length of vectors in word embedding.
        """

        # Load word embedding

        self.embedding_dim = embedding_dim

        self.embedding_matrix = load_word_embedding(embedding_path, self.word_index)

    def CNN(
        self,
        cnn_num_filters: int = 128,
        cnn_kernel_size: int = 5,
        cnn_activation: str = "relu",
        dropout: float = 0.4,
        trainable: bool = True,
    ):
        """ Simple Convolution Neural Network with regularisation

        Args:
            cnn_num_filters: Number of filters.
            cnn_kernel_size: CNN kernel size.
            cnn_activation: Activation of final layer.
            dropout: Amount of dropout to use in model.
        """

        word_input = tf.keras.Input(shape=(self.seq_length,))

        word_embedding = tf.keras.layers.Embedding(
            self.vocab_size + 1,
            self.embedding_dim,
            trainable=trainable,
            weights=[self.embedding_matrix],
            mask_zero=False,
        )(word_input)

        x = word_embedding
        inputs = word_input

        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(
            filters=cnn_num_filters,
            kernel_size=cnn_kernel_size,
            activation=cnn_activation,
        )(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def fit(
        self,
        epochs: int = 40,
        batch_size: int = 1024,
        loss_func: str = "binary_crossentropy",
        learning_rate: float = 0.01,
    ):
        """Fit the model

        Args:
            model_arch: Which model architecture to use. See
                TextClassifier.models for a complete list of possible
                architectures.
            epochs: Number of epochs to run model.
            batch_size: Batch size.
            loss_func: Loss function.
            learning_rate: Learning rate.
            cv: Is cross validation being used on this run? If set to `True`
                the checkpoint callback will be disabled, speeding up training
                time.
        """

        optimizer = optimizers.Adam(lr=learning_rate, clipnorm=1.0)

        self.model = self.CNN()

        self.model.compile(loss=loss_func, optimizer=optimizer, metrics=["accuracy"])

        self.model.summary()

        logger.info("Beginning training")

        self.history = self.model.fit(
            self.X_train_padded,
            self.y_train,
            epochs=epochs,
            validation_data=(self.X_test_padded, self.y_test),
            batch_size=batch_size,
            verbose=2,
        )

        self.model.save(os.path.join(self.model_output_path, "model.h5"))


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
    num_words=1000,
    oov_token="<OOV>",
    padding_style="pre",
    trunc_style="pre",
):
    # Create output path if not exists

    os.makedirs(output_path, exist_ok=True)

    # Set up logging

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh = logging.FileHandler(os.path.join(output_path, "log.txt"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Instantiate the model class

    CNN = TextClassifier(output_path=output_path, model_output_path=model_output_path)

    # Load the data from disk

    CNN.load_data(train_data_path, test_data_path)

    # Prepare the data with tokenisation, padding, etc.

    CNN.prep_data(
        oov_token=oov_token,
        trunc_style=trunc_style,
        padding_style=padding_style,
        num_words=int(num_words),
        lowercase=lowercase,
        save_tokenizer=True,
    )

    # Load word embedding from disk

    CNN.load_word_embedding(
        embedding_path=embedding_path, embedding_dim=int(embedding_dim),
    )

    # Fit the model

    CNN.fit(
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
    )


if __name__ == "__main__":
    app()
