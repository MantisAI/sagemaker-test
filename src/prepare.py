#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple training script
"""

from __future__ import annotations

import logging
import os
import pickle

import numpy as np
import yaml

import src.logger
import tensorflow as tf
import typer
from src.CNN import CNN
from src.load_word_embedding import load_word_embedding
from src.utils import read_jsonl
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

logger = logging.getLogger(__name__)

app = typer.Typer()


np.random.seed(1337)


params = yaml.safe_load(open("params.yaml"))


@app.command()
def prepare(
    test_path=params["common"]["test-path"],
    train_path=params["common"]["train-path"],
    data_path=params["common"]["all-data-path"],
    indices_path=params["common"]["indices-path"],
    output_path=params["train"]["output-path"],
    model_output_path=params["train"]["model-output-path"],
    lowercase=params["train"]["lowercase"],
    seq_length=params["train"]["seq-length"],
    num_words=params["train"]["num-words"],
    oov_token=params["train"]["oov-token"],
    padding_style=params["train"]["padding-style"],
    trunc_style=params["train"]["trunc-style"],
    test_prop=params["prepare"]["test-prop"],
):
    for path in [output_path, model_output_path, os.path.split(data_path)[0]]:
        os.makedirs(path, exist_ok=True)

    cnn = CNN(
        output_path=output_path,
        model_output_path=model_output_path,
        seq_length=int(seq_length),
    )

    # Load the data from disk

    cnn.load_data(data_path, test_prop)

    # Prepare the data with tokenisation, padding, etc.

    cnn.prep_data(
        oov_token=oov_token,
        trunc_style=trunc_style,
        padding_style=padding_style,
        num_words=int(num_words),
        lowercase=lowercase,
        save_tokenizer=True,
    )

    # Save the intermediate objects to disk

    cnn.save_indices(indices_path)

    # Save split data to disk as np arrays

    cnn.save_train_test_data(test_path, train_path)


if __name__ == "__main__":
    app()
