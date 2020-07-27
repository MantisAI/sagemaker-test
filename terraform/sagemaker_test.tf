
terraform {
  required_version = ">= 0.12"
  required_providers {
    aws      = "~> 2.56"
    dns      = "~> 2.2"
    template = "~> 2.1"
  }

  backend "local" {
  }
}

variable "region" {
  default = "eu-west-1"
}

provider "aws" {
  region  = var.region
}

#
# SageMaker ECR repo 
#

resource "aws_ecr_repository" "sagemaker_test" {
  name = "sagemaker-test"

}

#
# SageMaker Execution Role
#

module "sagemaker_execution_role" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-assumable-role"
  version = "~> 2.0"

  create_role = true

  role_name             = "sagemaker-exec-role"
  trusted_role_services = ["sagemaker.amazonaws.com"]

  custom_role_policy_arns = [
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess",
    "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
    "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess",
  ]

  role_requires_mfa = false
}

# S3 Bucket for use as DVC remote
#

module "sagemaker_test_s3" {
  source = "terraform-aws-modules/s3-bucket/aws"

  bucket_prefix = "sagemaker-test-"
  force_destroy = true

  versioning = {
    enabled = false
  }

}

#
# Outputs
#

output "sagameker_ecr_repo_name" {
  value = aws_ecr_repository.sagemaker_test.repository_url
}

output "sagameker_test_s3_bucket_name" {
  value = module.sagemaker_test_s3.this_s3_bucket_id
}

output "sagemaker_role_arn" {
  value = module.sagemaker_execution_role.this_iam_role_arn
}

