# DVC + MLFlow + Sagemaker training

This is an example project making use of three tools for managing machine learning workflows.

* [Data Version Control (DVC)](https://dvc.org/)
* [MLFlow](https://mlflow.org/)
* [AWS SageMaker](https://aws.amazon.com/sagemaker/)

The project itself is a tensorflow based deep learning project that classifies IMDB movie reviews as either good or bad. The data is downloaded from [Stanford University](https://ai.stanford.edu/~amaas/data/sentiment/). The project also makes use of 50 dimensional word embeddings also from [Stanford](https://nlp.stanford.edu/projects/glove/).

## Rationale

The rationale behind this project is as follows:

* DVC is awesome, and we should be using it version control our data in the same we we version control our code.
* We also want to store the results of our experiments, and ML Flow provides a way to do this with minimal friction, and no manual copying and pasting of results.
* Often training jobs are too big to run on our local machines, even though we may have done the data preparation locally. SageMaker allows us to run training jobs in the cloud in a cost effective way with access to GPUs as required.
* We still want to access the benefits of DVC and MLFlow when we are running cloud jobs, there shouldn't be a different process for running local or remote jobs.

Some things that I would like to include, but haven't implemented yet:

* Allow docker containers to be interrupted and restarted mid training. This will allow us to make use of managed spot instances on AWS, potentially reducing the compute cost by 90%.
* Model artefacts are not getting uploaded to MLFlow during training on SageMaker, although metrics and parameters are.

## Prerequisites

docker
docker-compose
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

If you wish to experiment with using your own s3 remote for your data, so you get the full DVC experience, you will need to add the s3 bucket name as a remote to `.dvc/config` as described [in the DVC documentation](https://dvc.org/doc/command-reference/remote/add).

## Getting started

### Run the pipeline locally

Clone the project, create a virtual environment and activate it:

```
make virtualenv
source build/virtualenv/bin/activate
```

Next you must launch an ML Flow ui with:

NOTE: The env var `MLFLOW_URI` should be set to `http://localhost:5000` to point to this server.

```
mlflow ui
```

This will create a server to collect your local experiment results. Ther server runs a web application at http://localhost:5000.

To run the dvc pipeline you will first need to get the word embedding. This can be done with dvc (Note that you need to have your AWS credentials accessible in this project either as `AWS_PROFILE` or with both `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` env vars set. If you don't have these set, you will experience a permissions error as this bucket is accessible only by authenticated AWS users).

```
dvc pull data/raw/glove.6B.50d.txt.dvc
```

To run the dvc pipeline from end to end:

```
dvc repro
```

Finally navigate to `http://localhost:5000` in a browser. You should be able to see the training job that just completed. You will be able to look at the model parameters and metrics that result from this training job.

### Update the pipeline artefacts on the remote

Now that the training job is completed, you should upload the model artefacts to remote storage. This is required for running a SageMaker job because the containers will download the data from s3 - the data is not packaged in the containers.

Update the `.dvc/config` file to reflect the following:

```
[core]
    remote = myremotebucket
['remote "muanalytics"']
    url = s3://muanalytics-public/sagemaker-test
['remote "myremotebucket"']
    url = s3://<bucket>/<folder>
```

Push the artefacts to your bucket with:

```
dvc push
```

### Changing paramaters

Try changing a paramater in `params.yaml`, for example the number of epochs, and then re-running the process with:

```
dvc repro
dvc push
```

### Running a job on SageMaker

In order to run a job on SageMaker you must build a docker container and push it the the Amazon Elastic Container Registry (ECR) repository that we created earlier with terraform.

There are two Dockerfiles provided here: one for CPU and one for GPU tasks. `Dockerfile.cpu` is a custom Dockerfile entirely for running tensorflow tasks on CPU. `Dockerfile.gpu` is based on a custom Dockerfile provided by Amazon for running GPU tasks on SageMaker. These images are relatively large, particularly the GPU version which weights in at some 5GB (there is probably scope for reducing this in future - but most of it comes from Amazon's base image).

Docker images can be built with docker-compose, but first we must make enviornment variables accessible to docker-compose by creating a `.env` file. We cannot pass environment variables to the containers at the point of running them, so some env vars must be hardcoded into the container at the point of build. The most important of these is the `MLFLOW_URI`. Obviously because we will be running the container on SageMaker, we must expose this API to the internet. The easiest way to do this is with `ngrok`.

After launching MLFLOW UI with `mflow ui` we can expose this to a url with `ngrok http 5000`. Ngrok will forward the port to a subdomain of `ngrok.io`, something like `http://fd4818758324.ngrok.io`. Copy this value from the ngrok command line tool into the `MLFLOW_URI` variable in `.envrc`.

NOTE: that a more sustainable way of doing this would be to set up a permanent MLFlow instance running on a remote instance or service, as in the present method we must rebuild the docker containers each time we change the ngrok address.

Make sure that you have also update the `REPO_URL` env var and `VERSION` if required. You can leave the `INSTANCE_TYPE` as local for now as we will test the job locally before attempting a run on SageMaker.

Once you are happy with the env variables set in `.envrc`, run `make .env` to populate these values into a `.env` file which will be read by docker-compose.

Create the docker images with:

```
sudo docker-compose build
```

We can now test the containers by running `python src/sagemaker_call.py --no-gpu --instance-type local`. Because the `instance-type` is set to local, this will execute a SageMaker job on the local machine.

NOTE: You may need to specify your python executable with: `build/virtualenv/bin/python src/sagemaker_call.py`.

Assuming all is well, we should try to execute a task on SageMaker itself. First we must push the containers we have built with:

```
make docker-login
sudo docker-compose push
```

Now run a remote task on SageMaker:

```
build/virtualenv/bin/python src/sagemaker_call.py --no-gpu --instance-type ml.c5.xlarge
```

To run a GPU task:

```
build/virtualenv/bin/python src/sagemaker_call.py --gpu --instance-type ml.p2.xlarge
```


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
