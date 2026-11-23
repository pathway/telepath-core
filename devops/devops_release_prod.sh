set -x

cat ../telepath_worker/version.txt

./devops_rebuild_docker.sh  # assuming you need to do that still
./devops_release_s3_push_prod.sh
./devops_push_ecr_prod.sh
