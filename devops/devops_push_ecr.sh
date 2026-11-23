set -x

app=$1 #telepath
version=$2
ecr_repo=$3  # telepath-dev
ecr_home=$4 # 123456789012.dkr.ecr.us-west-2.amazonaws.com

# region=us-west-2
# app=telepath
# ecr_repo=telepath-dev
# ecr_home=123456789012.dkr.ecr.$region.amazonaws.com

# get fresh login token
source ecr_login.sh

docker tag $app:latest $app:$version
docker tag $app:latest $ecr_home/$ecr_repo:latest
docker tag $app:$version $ecr_home/$ecr_repo:$version

# not needed
#docker push $ecr_home/$ecr_repo:latest
docker push $ecr_home/$ecr_repo:$version
