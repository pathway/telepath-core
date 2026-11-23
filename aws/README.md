https://fredhutch.github.io/aws-batch-at-hutch-docs/#what-is-aws-batch

https://aws.amazon.com/blogs/compute/creating-a-simple-fetch-and-run-aws-batch-job/


# Build a Docker image with the fetch & run script

...
ENTRYPOINT ["/usr/local/bin/fetch_and_run.sh"]
docker images

aws ecr get-login --region us-east-1 | sed 's/-e none //g'
### remove "-e none" from result before running
### AWS Account ID: 123456789012


# SETUP:1time: Create an Amazon ECR repository for the image

# SETUP:1time:  Create an IAM role to be used by jobs to access S3
- https://console.aws.amazon.com/iam/home?region=us-east-1#security_credential
- You need an IAM role that the AWS Batch job can use to access S3
    - Use case : 
- Hacky way is to include keys directly

# SETUP:1time:  Create a simple job script and upload it to S3




# SETUP:per_image: Push the built image to ECR
docker tag nvidia/cuda:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/nvidia/cuda:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/nvidia/cuda:latest



docker build -t telepath .  # test
docker container run --gpus all telepath:latest # test
docker tag telepath:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/telepath:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/telepath:latest

# 123456789012.dkr.ecr.us-east-1.amazonaws.com/telepath1:latest

# SETUP:per_image: Create a job definition that uses the built image

###
- https://console.aws.amazon.com/batch/home?region=us-east-1#/job-definitions

- revised to: https://console.aws.amazon.com/batch/home?region=us-east-1#/job-definitions/arn:aws:batch:us-east-1:123456789012:job-definition~2Fjob_defn_01:2
- job_defn_01:2



# RUN: Submit and run a job that execute the job script from S3
- see batch_submit.py

### 
- https://console.aws.amazon.com/batch/home?region=us-east-1#/dashboard
- Create job
- Job definition: job_defn_01:2
- Job queue: 
- "An array job has children that are replicas of the parent and all have access to the parameters passed in."
- Job depends on: 
- Resource requirements: GPUs: 1
- Job status: SUBMITTED, RUNNABLE, 
- Show full job logs in cloudwatch


