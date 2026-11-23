# Syscheck

Syscheck is a test container designed to confirm GPU access, ECR access, S3 access, etc.

## Elastic Container Registry

Syscheck is run via docker, and can be accessed via AWS Elastic Container Registry:

    ARN     123456789012.dkr.ecr.us-west-2.amazonaws.com/syscheck-01
    Region  us-west-2
    Tag     latest

## Usage

Check deep learning + GPU environment:
```
python3 /src/syscheck.py deepcheck
```

Check S3 access:
```
python3 /src/syscheck.py s3check BUCKET KEY
```
- this will (destructively) test ability to write to and read from _s3://bucket/key_


## Telepath container

Syscheck is also included in the telepath image, and can be run from there using docker command lines like this:
```
/src/telepath-core/syscheck deepcheck
/src/telepath-core/syscheck s3check BUCKET KEY
```
