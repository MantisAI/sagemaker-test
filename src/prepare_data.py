import logging
import os
import random

from src.utils import write_jsonl
from tqdm import tqdm

logger = logging.getLogger(__name__)

source = {
    "train": ["data/raw/aclImdb/train/neg", "data/raw/aclImdb/train/pos"],
    "test": ["data/raw/aclImdb/test/neg", "data/raw/aclImdb/test/pos"],
}


def combine_data(input_file_list, base_path, label):
    """Iterate through a list of files add them to a single file and save to
    jsonl
    """
    out = []

    for file in tqdm(input_file_list):
        path = os.path.join(base_path, file)
        with open(path, "r") as fb:
            content = fb.read()
            out.append(
                {"text": content, "label": label,}
            )

    return out


if __name__ == "__main__":

    os.makedirs("data/processed", exist_ok=True)

    for k, v in source.items():
        logger.info("Processing %s", k)
        out = []  # type: list

        for dir in v:
            files = os.listdir(dir)
            label = os.path.basename(dir)

            out.extend(combine_data(files, dir, label))
        random.shuffle(out)
        write_jsonl(out, f"data/processed/{k}.jsonl")
