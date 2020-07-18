# -*- coding: utf-8 -*-
"""Multiclass Multilabel model for labelling comments/tweets with picker
    targets
"""

import logging

import mnist_cnn
import typer

logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def foo():
    return None


@app.command()
def train():

    mnist_cnn.run()


if __name__ == "__main__":
    app()
