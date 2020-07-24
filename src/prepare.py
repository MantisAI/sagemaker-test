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
import yaml
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import src.logger
from src.CNN import CNN
from src.load_word_embedding import load_word_embedding
from src.utils import read_jsonl

logger = logging.getLogger(__name__)

app = typer.Typer()


np.random.seed(1337)



@app.command()
def prepare(
    data_path="data/intermediate/data.jsonl",
    output_path="models",
    model_output_path="models",
    lowercase=True,
    seq_length=1000,
    num_words=1000,
    oov_token="<OOV>",
    padding_style="pre",
    trunc_style="pre",
    test_prop=0.3,
    processed_path="data/processed",
):
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

    cnn.save_indices()

    # Save split data to disk as np arrays

    cnn.save_train_test_data(processed_path)


if __name__ == "__main__":
    app()
