set -e
set -x

tag=basetest3
region=us-west-2
ecr_url=123456789012.dkr.ecr.$region.amazonaws.com

# Login before running this:
# ---
### aws ecr get-login-password --region $region |   docker login --username AWS --password-stdin $ecr_url
### aws ecr get-login --region $region | sed 's/-e none //g'

#rm Dockerfile
#ln -s Dockerfile.$tag Dockerfile

## test the scripts: run python tests before building
# pytest

cd ..

docker build -t $tag:latest .

## test the image
#docker container run --gpus all basetest1:latest python3 telepath.py status --config=config/cartpole.yaml
## check for success output

docker tag $tag:latest $ecr_url/$tag:latest
docker push $ecr_url/$tag:latest


# -- Part 2
# separate this into another script
# push

cd aws/
python telepath_ctl.py list_jobs
python telepath_ctl.py launch  fulltest # command_file.yaml
python telepath_ctl.py list_jobs

# Add 2FA with aws

cd devops
