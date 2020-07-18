import os

from sagemaker.estimator import Estimator

HOST = os.getenv("HOST")
REPO = os.getenv("REPO")
VERSION = os.getenv("VERSION")

ROLE_ARN = os.getenv("ROLE_ARN")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE")

estimator = Estimator(
    image_name=f"{HOST}/{REPO}:{VERSION}",
    role=ROLE_ARN,
    train_instance_count=1,
    train_instance_type=INSTANCE_TYPE,
)

estimator.fit()
