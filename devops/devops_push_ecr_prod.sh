set -x

version=`cat ../telepath_worker/version.txt`

./devops_push_ecr.sh telepath $version telepath 123456789012.dkr.ecr.us-west-2.amazonaws.com
