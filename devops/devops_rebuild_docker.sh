set -e
set -x

region=us-west-2
version=`cat ../telepath_worker/version.txt`

# versioning: see https://medium.com/better-programming/how-to-version-your-docker-images-1d5c577ebf54

# test running locally
#cd telepath_worker
#python ./telepath status --config=specs2/telepath_settings.yaml

cd ..

docker build -t telepath:$version .

# export AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID_PLACEHOLDER
# export AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY_PLACEHOLDER

# test it
# docker container run --gpus all telepath:latest python3 telepath.py status --config=specs2/telepath_settings.yaml
# docker container run --gpus all telepath:latest python3 telepath.py fulltest --config=specs2/telepath_settings.yaml

# docker container run --gpus all -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY telepath:latest python3 telepath.py fulltest --config=specs2/telepath_settings.yaml

#enter it
#docker container run --gpus all -it telepath:latest  bash
docker container run --gpus all -it telepath:$version telepath status --project_spec=specs/client_sample/spec_project.yaml

cd devops

#./devops_push_ecr.sh telepath $version telepath-dev 123456789012.dkr.ecr.us-west-2.amazonaws.com

# app=telepath
# ecr_repo=telepath-dev
# ecr_home=123456789012.dkr.ecr.$region.amazonaws.com

# docker tag telepath:latest telepath:$version
# docker tag telepath:latest 123456789012.dkr.ecr.$region.amazonaws.com/telepath:latest
# docker tag telepath:$version 123456789012.dkr.ecr.$region.amazonaws.com/telepath:$version

# docker push 123456789012.dkr.ecr.$region.amazonaws.com/telepath:latest
# docker push 123456789012.dkr.ecr.$region.amazonaws.com/telepath:$version

# cd aws/
# python telepath_batch.py list_jobs
# python telepath_batch.py launch
# python telepath_batch.py list_jobs
