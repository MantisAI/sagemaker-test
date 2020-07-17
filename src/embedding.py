""" Download the word embedding and extract
"""

import logging
import os
import zipfile

import requests
import yaml

import src.logger
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    params = yaml.safe_load(open("params.yaml"))["embedding"]

    logger.info("Downloading %s", params["we-url"])

    zip_path = params["we-archive-path"]
    response = requests.get(params["we-url"], stream=True)

    ### Download with a progress bar

    with tqdm.wrapattr(
        open(zip_path, "wb"),
        "write",
        miniters=1,
        total=int(response.headers.get("content-length", 0)),
        desc=zip_path,
    ) as fout:

        for chunk in response.iter_content(chunk_size=4096):
            fout.write(chunk)

    logger.info(
        "Extracting %s from %s to %s",
        os.path.basename(params["we-path"]),
        zip_path,
        params["we-path"],
    )

    with zipfile.ZipFile(zip_path, "r") as fd:
        fd.extract(
            member=os.path.basename(params["we-path"]),
            path=os.path.split(params["we-path"])[0],
        )

    logger.info("Wrote embedding to %s", params["we-path"])
