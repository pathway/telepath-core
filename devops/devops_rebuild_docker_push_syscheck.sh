set -e
set -x

region=us-west-2
tag=syscheck
ecr_repo=syscheck-01

cd ..
docker build -f Dockerfile.syscheck -t $tag .

# test it
docker container run --gpus all -it $tag:latest  python3 syscheck.py deepcheck
docker container run --gpus all -it $tag:latest  python3 syscheck.py s3check bucket key

#docker container run --gpus all $tag:latest python3 telepath.py status --config=config/cartpole.yaml
#docker container run --gpus all -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY $tag:latest python3 telepath.py fulltest --config=config/docker_cartpole_s3.yaml

#enter it
docker container run --gpus all -it $tag:latest  bash


# push
docker tag $tag:latest 123456789012.dkr.ecr.$region.amazonaws.com/$ecr_repo:latest

docker push 123456789012.dkr.ecr.$region.amazonaws.com/$ecr_repo:latest

cd devops
