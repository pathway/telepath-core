ver=`cat ../telepath_worker/version.txt`
docker build -t telepath . && docker container run --gpus all telepath:$ver

docker container run --gpus all telepath$ver python3 telepath.py status --config=config/docker_cartpole.yaml
docker container run --gpus all telepath$ver python3 telepath.py fulltest --config=config/docker_cartpole.yaml

docker container run --gpus all telepath$ver nvidia-smi


docker container run  -e AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID_PLACEHOLDER -e AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY_PLACEHOLDER \
     --gpus all telepath$ver \
      python3 telepath.py fulltest --config=config/docker_cartpole_s3.yaml

