from setuptools import setup
import os

setup(
    name="sagemaker_example",
    packages=["src"],
    version=os.getenv("VERSION"),
    description="SageMaker example",
    author="Matthew Upson",
    author_email="matt@muanalytics.co.uk",
    license="MIT",
)
