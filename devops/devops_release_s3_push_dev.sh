
version=`cat ../telepath_worker/version.txt`-dev
bucket=telepath-client-01-dev

./devops_release_s3_push.sh $version $bucket
