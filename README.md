# DVC + MLFlow + Sagemaker training

This is an example project making use of three tools for managing machine learning workflows.

* [Data Version Control (DVC)](https://dvc.org/)
* [MLFlow](https://mlflow.org/)
* [AWS SageMaker](https://aws.amazon.com/sagemaker/)

The project itself is a tensorflow based deep learning project that classifies IMDB movie reviews as either good or bad. The data is downloaded from [Stanford University](https://ai.stanford.edu/~amaas/data/sentiment/). The project also makes use of 50 dimensional word embeddings also from [Stanford](https://nlp.stanford.edu/projects/glove/).

## Prerequisites

Python 3.6.6+
make
ngrok
direnv (or similar for handling env vars)
terraform
An AWS Account

## Some Infrastructure

You will need the following bits of AWS infrastructure:

* An S3 Bucket
* A ECR repository
* An execution role for use with SageMaker

Terraform manifests for managing this infrastructure are included in `./terraform`, but youc an also create it using the AWS console or CLI.

To create the infrastructure:

```
cd ./terraform
terraform init
terraform apply # type yes when prompted
```

The terraform creation will finished with some output values which will be required later, e.g.:

```
Outputs:

sagameker_ecr_repo_name = 111111111111.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-test
sagameker_test_s3_bucket_name = sagemaker-test-20200727213242375500000001
sagemaker_role_arn = arn:aws:iam::111111111111:role/sagemaker-exec-role
```

If using direnv, you can use these outputs to populate the `REPO_URL` and `ROLE_ARN` env vars in the `.envrc` file.

## Workflow

The project uses a DVC pipeline to download, extract and prepare the data, and then train a CNN using tensorflow. DVC creates a Directed Acyclic Graph (DAG) which determines which component needs to be run, so it will not repeat tasks which do not need to be run.

Each experiment is monitored with ML Flow which maintains a database of experiments, results, and parameters.

The project can be run locally or on AWS SageMaker (including on GPU) using custom built docker containers which are deployed to Amazon Elastic Compute Repository (ECR).

## Getting started

### Get the data

Clone the project and create a virtual environment and activate it:

```
make virtualenv
source build/virtualenv/bin/activate
```

To download all the data required to run the project (this will download all the data and model outputs from a public s3 bucket specified in `.dvc/config`):

NOTE: You will need to have an AWS account set up with the necessary credentials available in this project. This may mean setting up youe account with `aws config` using awscli, or setting your `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in your project env vars. Note that this S3 bucket is set to 'requester pays', so you will be charged for the data transfer fees (an insignificant amount!).

```
dvc pull
```

At this point, you should probably create your own s3 bucket to house a dvc remote. Once you have created this, change the remote listed in the `.dvc/config` file to match your own bucket. This will allow you to have full DVC remote functionality, which would otherwise be impossible since the current s3 bucket is read only. Use the s3 bucket name that was output by terraform, or another bucket that you have access to.

Once you have edited `./dvc/config` you can push the data to your remote with `dvc push`.

### Set up MLFlow

MLFlow runs a server and web application that records metrics, and parameters from training runs (amongst other things). To launch a local instance:

```
source build/virtualenv/bin/activate
mlflow ui
```

The env var `MLFLOW_URI` should be set to `http://localhost:5000` to point to this server.

To run the model locally, modify one of the parameters in `params.yaml`, for example change the number of epochs, or the train




## Resources

* [How to make Docker containers with Amazon Sagemaker Container Library](https://docs.aws.amazon.com/sagemaker/latest/dg/amazon-sagemaker-containers.html)
* [Pre-built SageMaker containers](https://github.com/aws/deep-learning-containers)
* [DEPRECATED - Additional guidance on building SageMaker containers](https://github.com/aws/sagemaker-containers)
* [Complete list of env vars](https://github.com/aws/sagemaker-training-toolkit/blob/master/src/sagemaker_training/params.py). NOTE: above item contains better documentation for these.
* [Source code - probably the best palce to understand the class](https://github.com/aws/sagemaker-python-sdk/blob/1872483681a6647bdad126b8214fb6cc35e164fd/src/sagemaker/estimator.py#L1133)
* [Instructions for using spot instances](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html)
* [Estimator class API](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html)


## data

Downloaded from http://ai.stanford.edu/~amaas/data/sentiment/ with:

```
dvc get-url http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz data/
```

If you are seeing: 

```
botocore.exceptions.ClientError: An error occurred (ValidationException) when calling the CreateTrainingJob operation: No S3 objects found under S3 URL "s3://muanalytics-dvc/sagemaker-test/09/fcd808fea1330310e94e430e4bd0d2" given
in input data source. Please ensure that the bucket exists in the selected region (eu-west-1), that objects exist under that S3 prefix, and that the role "arn:aws:iam::203149375586:role/sagemaker-execution-role" has "s3:ListBucket" permissions on bucket "muanalytics-dvc".
```

Try `dvc push` as it is possible that the version of the input files is out of date on the remote (muanalytics-dvc)]u
