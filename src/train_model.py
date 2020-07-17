# -*- coding: utf-8 -*-
"""Multiclass Multilabel model for labelling comments/tweets with picker
    targets
"""

import json
import os
from datetime import datetime

import typer
from wasabi import msg
from src.logger import logger

from src.models.text_classifier import TextClassifier, results_formatter
from src.utils.get_file_from_s3 import get_file_from_s3


def train_model(
    config: str, epochs: int = None, kfolds: int = None, test_prop: float = 0.2
):

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load json config:

    with open(config, "r") as fb:
        cfg = json.load(fb)

    if epochs:
        EPOCHS = epochs
    else:
        EPOCHS = cfg["epochs"]

    if kfolds:
        KFOLDS = kfolds
    else:
        KFOLDS = cfg["kfolds"]

    if test_prop:
        TEST_PROP = test_prop
    else:
        TEST_PROP = cfg["test_prop"]

    # Check for word embedding and download if not exists

    get_file_from_s3(cfg["embedding_path"], cfg["embedding_s3_url"])

    basename = os.path.basename(os.path.split(cfg["train_data"])[1])
    picker_path = os.path.split(cfg["train_data"])[0]

    train_path = os.path.join(picker_path, f"{basename}_train.npz")
    test_path = os.path.join(picker_path, f"{basename}_test.npz")

    # Create the model

    picker = TextClassifier(
        output_path=cfg["output_path"],
        class_names=cfg["targets"],
        char_embedding=cfg["char_embedding"],
        version=cfg["version"],
    )

    # Only need to create the test train split once, it can then be loaded from
    # npz files

    try:
        picker.load_train_test_data(path=picker_path, basename=basename)
        msg.good(
            f"Found existing train/test split data {os.path.join(picker_path, basename)}"
        )
    except FileNotFoundError:
        # TODO: better error handling here
        msg.good(f"No train/test split found, creating...")
        picker.load_data(cfg["train_data"], test_prop=test_prop)
        picker.save_train_test_data(path=picker_path, basename=basename)
    #    except Exception:
    #        logger.exception(f"Unhandled exception running f{cfg['version']}")

    picker.prep_data(
        oov_token=cfg["oov_token"],
        trunc_style=cfg["trunc_style"],
        padding_style=cfg["padding_style"],
        digits_token=cfg["digits_token"],
        num_words=cfg["num_words"],
        lowercase=cfg["lowercase"],
        save_indices=True,
    )

    picker.load_word_embedding(
        embedding_path=cfg["embedding_path"], embedding_dim=cfg["embedding_dim"],
    )

    picker.fit(
        model_arch=cfg["model"],
        epochs=EPOCHS,
        batch_size=cfg["batch_size"],
        loss_func="binary_crossentropy",
        learning_rate=cfg["learning_rate"],
    )

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results = picker.evaluate(target_names=picker.class_names)

    results_formatter(
        config=config,
        output_path=cfg["output_path"],
        summary_name="single_result_summary.csv",
        results=results,
        start=start_time,
        end=end_time,
    )

    picker.load_model(os.path.join(cfg["output_path"], "model.h5"))


if __name__ == "__main__":
    typer.run(train_model)
