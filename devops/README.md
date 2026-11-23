Top level:

- development
```
# build docker and try pushing to dev ECR
./devops_rebuild_docker_push.sh # pushes to dev repo only
```

- production
```
# release ECR and S3
./devops_release_prod.sh
```

- ECR only
```
# release to ECR only
./devops_push_ecr_prod.sh
./devops_push_ecr_dev.sh
```
