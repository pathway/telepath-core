# from https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html
# ---
# docker build -t hello-world .
# aws ecr create-repository --repository-name hello-repository --region region
# docker tag hello-world aws_account_id.dkr.ecr.region.amazonaws.com/hello-repository
# aws ecr get-login-password | docker login --username AWS --password-stdin aws_account_id.dkr.ecr.region.amazonaws.com
# docker push aws_account_id.dkr.ecr.region.amazonaws.com/hello-repository
# aws ecr delete-repository --repository-name hello-repository --region region --force  # cleanup

# from https://us-east-1.console.aws.amazon.com/ecr/repositories?region=us-east-1
# ---

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
docker build -t telepath1 .
docker tag telepath:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/telepath:latest

