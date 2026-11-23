# Amazon Web Services

## Instance types 
Suggested instance types:

- testing: p2.xlarge
- production: g4dn.8xlarge

<br>

## Elastic Container Registry
Telepath docker image can be accessed via AWS Elastic Container Registry:

    ARN     123456789012.dkr.ecr.us-west-2.amazonaws.com/telepath
    Region  us-west-2

<br>

## S3
- Telepath is compatible with AWS S3 for file storage
- all file types (including training data, spec files, agent files) can be refered to by local path, or s3:// url
- of course, this assumes your process has the needed s3 permissions for the bucket
