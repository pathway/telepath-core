set -x

cat ../telepath_worker/version.txt

./devops_release_s3_push_dev.sh
./devops_push_ecr_dev.sh
