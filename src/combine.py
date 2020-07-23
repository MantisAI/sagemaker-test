""" Cycle through the files we want into the tar and combine into a jsonl
file
"""
import logging
import os
import random
import re
import tarfile

import yaml

from src.utils import write_jsonl
import src.logger
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Get params from params file.

params = yaml.safe_load(open("params.yaml"))


if __name__ == "__main__":

    os.makedirs(os.path.split(params["common"]["all-data-path"])[0], exist_ok=True)

    out = []

    logger.info("Reading data form %s", params["common"]["download-path"])
    tar = tarfile.open(params["common"]["download-path"], "r:gz")

    for member in tqdm(tar.getmembers()):
        if re.match(r".*(train|test)\/(pos|neg).*", member.name):
            label = os.path.split(member.name)[0].split("/")[-1]
            file = tar.extractfile(member)

            if file is not None:
                logger.debug("Reading contents of %s", member.name)
                content = file.read().decode("utf-8")
                out.append(
                    {
                        "text": content,
                        "label": params["combine"]["label-mapping"][label],
                    }
                )

    random.shuffle(out)
    write_jsonl(out, params["common"]["all-data-path"])
    logger.info("Wrote data to %s", params["common"]["all-data-path"])
