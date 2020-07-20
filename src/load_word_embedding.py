#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from src.logger import logger


def load_word_embedding(embedding_path, word_index):

    """
    Convert word embeddings to dictionary
    Will return dict of the format: `{word: embedding vector}`.
    Only return words of interest. If the word isn't in the embedding, returns
    a zero-vector instead.
    Args:
        embedding_path(str): Path to embedding (which will be loaded).
        word2ind(dict): Dictionary {words: index}. The keys represented the
            words for each embeddings will be retrieved.
    Returns:
        Array of embeddings vectors. The embeddings vector at position i
        corresponds to the word with value i in the dictionary param `word2ind`
    """

    # Read the embeddings file

    embeddings_index = {}
    with open(embedding_path, "r") as file:
        logger.info(f"Reading embedding from {embedding_path}")
        for i, line in enumerate(file):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
            if i == 0:
                coef_length = len(coefs)

    # Compute the embedding for each word in the dataset

    embeddings_matrix = np.zeros((len(word_index) + 1, coef_length))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector

    # Drop the embeddings index from memory and just return the shrunken matrix

    del embeddings_index

    return embeddings_matrix
