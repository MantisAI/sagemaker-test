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
