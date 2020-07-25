#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple training script
"""

import logging
import os
import pickle
import random

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

from src.load_word_embedding import load_word_embedding
from src.logger import logger
from src.utils import read_jsonl

app = typer.Typer()


np.random.seed(1337)


class CNN:
    def __init__(
        self, output_path: str, model_output_path: str, seq_length: int = 1000,
    ):
        """A binary classification model based on a simple CNN

        Args:
            output_path: Path to dir where model results will be saved. Will be
                created if not exists.
            model_output_path: Path to dir where model object will be saved.
                Will be created if not exists.
            seq_length: Length of tokens to be fed to the model as training
                examples. Anything longer than this will be truncated. Anything
                shorter will be padded.
        """

        self.output_path = output_path
        self.model_output_path = model_output_path
        self.seq_length = seq_length
        self.embedding_dim = 0  # type: int
        self.history = None  # type: tf.keras.model.fit
        self.callbacks = []  # type: list

    def load_data(self, data_path: str, test_prop: float):
        """Load data from jsonl file

        Args:
            data_path: Path to jsonl file.
        """

        all_data = read_jsonl(data_path)

        random.shuffle(all_data)
        test_index = round(len(all_data) * test_prop)

        X_train, X_test, y_train, y_test = [], [], [], []

        for review in all_data[:test_index]:
            X_test.append(review["text"])
            y_test.append(review["label"])

        for review in all_data[test_index:]:
            X_train.append(review["text"])
            y_train.append(review["label"])

        self.y_test = np.array(y_test).astype(int)
        self.y_train = np.array(y_train).astype(int)
        self.X_test = np.array(X_test)
        self.X_train = np.array(X_train)

        logger.info("Train data shape: %s, %s", len(self.X_train), len(self.y_train))
        logger.info("Test data shape: %s, %s", len(self.X_test), len(self.y_test))

    def save_indices(self, indices_path: str):
        """Save model artefacts to pickle

        Saves various model artefacts to a pickle file so that they can be
        loaded at a later date for predictions.

        Always saves to a default location: output_path + indices.pickle
        """
        indices = {}
        indices["word_index"] = self.word_index
        indices["vocab_size"] = self.vocab_size
        indices["tokenizer"] = self.tokenizer
        with open(indices_path, "wb") as f:
            pickle.dump(indices, f)

    def load_indices(self, indices_path: str):
        """Loads model artefacts from pickle

        Companion method to `save_indices()`. Loads an indices.pickle object
        that has been saved with the `save_indices()`.
        """

        with open(indices_path, "rb") as f:
            indices = pickle.load(f)
        self.word_index = indices["word_index"]
        self.vocab_size = indices["vocab_size"]
        self.tokenizer = indices["tokenizer"]
        self.word_index = self.tokenizer.word_index

    def save_train_test_data(self, test_path: str, train_path: str):
        """Saves train and test data to disk as numpy arrays

        This is to ensure that a consistent dataset is used for single model
        runs, and is unrelated to cross validation.

        Args:
            train_path: Path to training npz
            test_path: Path to test npz
        """

        np.savez(train_path[:-4], x=self.X_train_padded, y=self.y_train)
        logger.info(f"Train data saved to {train_path}")

        np.savez(test_path[:-4], x=self.X_test_padded, y=self.y_test)
        logger.info(f"Train data saved to {test_path}")

    def load_train_test_data(self, test_path: str, train_path: str):
        """Load data from pre-split numpy arrays

        Loads data from numpy arrays that were saved using the
        save_train_test_data() method. Args should match between the two
        methods.

        Args:
            train_path: Path to training npz
            test_path: Path to test npz
        """

        train = np.load(train_path, allow_pickle=True)
        logger.info(f"Train data loaded from {train_path}")

        test = np.load(test_path, allow_pickle=True)
        logger.info(f"Test data loaded from {test_path}")

        self.X_train_padded, self.y_train = train["x"], train["y"]
        self.X_test_padded, self.y_test = test["x"], test["y"]

        logger.info(f"Train data shape: {len(self.X_train_padded)} {len(self.y_train)}")
        logger.info(f"Test data shape: {len(self.X_test_padded)} {len(self.y_test)}")

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

    def load_callbacks(
        self,
        early_stopping: bool = True,
        early_stopping_patience: int = 5,
        checkpoint: bool = True,
        checkpoint_path: str = "checkpoint",
    ):

        """ Prepare callbacks for model.fit

        Args:
            early_stopping: Use early_stopping callback?
            early_stopping_patience: Number of epochs that show now improvements
                before early stopping is activated.
            checkpoint: Use checkpointer callback?
            checkpoint_path: Path to where checkpoints will be saved.
        """

        # early_stopping = tf.keras.callbacks.EarlyStopping(
        #    monitor="val_loss",
        #    patience=early_stopping_patience,
        #    verbose=1,
        #    mode="auto",
        # )

        # self.callbacks.append(early_stopping)

        # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        #    monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001
        # )

        # self.callbacks.append(reduce_lr)

        if checkpoint:

            checkpointer = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, verbose=1, save_best_only=True,
            )

            self.callbacks.append(checkpointer)

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
            callbacks=self.callbacks,
        )

        # Don't save the model, this is handled by the checkpointer
        # tf.saved_model.save(self.model, self.model_output_path)
