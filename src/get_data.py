import gzip
import os

import requests
from tqdm import tqdm

s3_slug = "https://datalabs-public.s3.eu-west-2.amazonaws.com/deep_reference_parser/"

data = {
    "train": "data/multitask/2020.3.19_multitask_train.tsv",
    "test": "data/multitask/2020.3.19_multitask_test.tsv",
    "valid": "datamultitask/2020.3.19_multitask_valid.tsv",
}

os.makedirs("data", exist_ok=True)

for k, url in tqdm(data.items()):
    with open(f"data/{k}.tsv", "wb") as fb:
        r = requests.get(f"{s3_slug}/{url}", allow_redirects=True)
        fb.write(r.content)
