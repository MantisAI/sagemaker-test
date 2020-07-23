""" Download the data .tar.gz
"""

import logging
import os

import requests
import yaml

import src.logger
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    params = yaml.safe_load(open("params.yaml"))

    logger.info("Downloading %s", params["download"]["data-url"])

    gz_path = os.path.join(params["common"]["download-path"])
    response = requests.get(params["download"]["data-url"], stream=True)

    ### Download with a progress bar

    with tqdm.wrapattr(
        open(gz_path, "wb"),
        "write",
        miniters=1,
        total=int(response.headers.get("content-length", 0)),
        desc=gz_path,
    ) as fout:

        for chunk in response.iter_content(chunk_size=4096):
            fout.write(chunk)

    logger.info("Wrote data to %s", gz_path)
