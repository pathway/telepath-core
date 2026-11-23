set -x
region=us-west-2
rm _ecr_login_token_script.sh || true
aws ecr get-login --region $region | sed 's/-e none //g'  > _ecr_login_token_script.sh
source _ecr_login_token_script.sh

