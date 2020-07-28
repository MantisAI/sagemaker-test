""" Download and extract the word embedding
"""

import logging
import os
import zipfile

import requests
import yaml

import src.logger
from tqdm import tqdm
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    params = yaml.safe_load(open("params.yaml"))["embedding"]

    logger.info("Downloading %s", params["we-url"])

    zip_path = params["we-archive-path"]
    we_path = params["we-path"]
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

    logger.info("Wrote data to %s", zip_path)

    os.makedirs(os.path.split(zip_path)[0], exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        with open(we_path, "wb") as fd:
            fd.write(zf.read(os.path.split(we_path)[1]))

    logger.info("Wrote embedding to %s", we_path)
