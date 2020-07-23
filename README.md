# README

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

Try `dvc push` as it is possible that the version of the input files is out of date on the remote (muanalytics-dvc)
