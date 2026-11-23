set -e
set -x

version=$1
bucket=$2

cd ../docs/telepath_docs/
mkdocs build
cd ../../

echo " -- Replacing version string: "
sed -i "s/TELEPATH_VERSION/v$version/" docs/telepath_docs/site/*.html

echo " -- Publishing docs: "
aws s3 cp --recursive docs/telepath_docs/site  s3://$bucket/$version/docs
aws s3 cp --recursive docs/telepath_docs/site  s3://$bucket/latest/docs

echo " -- Publishing specs: "
aws s3 cp --recursive telepath_worker/specs  s3://$bucket/$version/specs
aws s3 cp --recursive telepath_worker/specs  s3://$bucket/latest/specs

echo " -- Local replica: "
mkdir -p  /tmp/telepath_docs/$version/
aws s3 cp --recursive s3://$bucket/$version/ /tmp/telepath_s3/$version/

ls /tmp/telepath_s3/$version/
echo "file:///tmp/telepath_docs/$version/"
cd devops
