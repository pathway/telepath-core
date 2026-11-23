# Telepath Controller

Controller is a command-line app, that uses the AWS API to bring up Telepath workers.  It coordinates of training on multiple containers.  

It uses the AWS Batch API to launch and monitor Telepath Worker jobs.  This makes it easy to train multiple policies at once.

## Settings
Controller is configured with a controller.yaml file:

    ---
    aws_region: 'us-west-2'
    jobName: 'telepath-job-001'
    jobQueue: 'arn:aws:batch:us-west-2:XXXXARNXXXXXX'
    jobDefinition: 'arn:aws:batch:us-west-2::XXXXARNXXXXXX'
