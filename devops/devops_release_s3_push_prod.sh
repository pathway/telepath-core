
version=`cat ../telepath_worker/version.txt`
bucket=telepath-client-01

./devops_release_s3_push.sh $version $bucket
