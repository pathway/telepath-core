# Integration test script generator

The Telepath container can help you construct integration testing scripts, so you can exercise the different telepath commands in sequence for testing.

Run the `integ_gen` script to generate telepath command line scripts.

For example:
```
integ_gen \
  --spec_root=s3://client-test-gen/test1/spec \  # spec folder, can be on s3
  --data_root=s3://client-test-gen/test1/data \ # read/write data folder, can be on s3
  --env_id=SimpleGym_Sequential_Short-v0 \  # which gym
  --exec_prefix="docker container run --gpus all -it telepath:0.8.1.10" # optional; incl version
```

This will generate output:
```
# spec_root s3://client-test-gen/test1/spec
# data_root s3://client-test-gen/test1/data
# env_id SimpleGym_Sequential_Short-v0
# exec_prefix docker container run --gpus all -it telepath:0.8.1.10
#
docker container run --gpus all -it telepath:0.8.1.10 ./telepath generate_replay --project_spec=s3://client-test-gen/test1/spec/spec_project.yaml --env_id=SimpleGym_Sequential_Short-v0 --out_replay=s3://client-test-gen/test1/data/replay_rawtrain.csv --replay_spec=s3://client-test-gen/test1/spec/spec_replay.yaml --algo_spec=s3://client-test-gen/test1/spec/spec_algo.yaml --agent_dest=s3://client-test-gen/test1/data/online.agent_telepath
#
docker container run --gpus all -it telepath:0.8.1.10 ./telepath prep --project_spec=s3://client-test-gen/test1/spec/spec_project.yaml --in_replay=s3://client-test-gen/test1/data/replay_rawtrain.csv --in_replay_spec=s3://client-test-gen/test1/spec/spec_replay.yaml --out_replay=s3://client-test-gen/test1/data/replay.csv --out_replay_spec=s3://client-test-gen/test1/spec/spec_replay.yaml --prep_spec=s3://client-test-gen/test1/spec/spec_replay_prep.yaml
#
docker container run --gpus all -it telepath:0.8.1.10 ./telepath train --project_spec=s3://client-test-gen/test1/spec/spec_project.yaml --replay=s3://client-test-gen/test1/data/replay.csv --replay_spec=s3://client-test-gen/test1/spec/spec_replay.yaml --algo_spec=s3://client-test-gen/test1/spec/spec_algo.yaml --agent_path=s3://client-test-gen/test1/data/batch.agent_telepath
#
docker container run --gpus all -it telepath:0.8.1.10 ./telepath predict --project_spec=s3://client-test-gen/test1/spec/spec_project.yaml --in_features_path=s3://client-test-gen/test1/data/replay.csv --in_spec=s3://client-test-gen/test1/spec/spec_replay.yaml --agent_path=s3://client-test-gen/test1/data/batch.agent_telepath --out_pred_path=s3://client-test-gen/test1/data/predict.csv

```

This simplifies the writing of the telepath commands.
