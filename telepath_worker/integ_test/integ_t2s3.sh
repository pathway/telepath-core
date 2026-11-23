#set -e
#set -x

dir_root=s3://example-telepath-bucket/t2

proj_spec=$dir_root/specs2/telepath_settings_s3.yaml
replay_path=$dir_root/tmp/replay.csv
#replay_spec=$dir_root/specs2/spec_replay_cartpole_tiny.yaml
replay_spec=$dir_root/specs2/spec_replay_simplegym_seq_short.yaml

algo_spec=$dir_root/specs2/tiny_spec_algo.yaml

agent_folder=$dir_root/tmp/
online_agent_folder=$dir_root/tmp/
online_agent_path=$dir_root/tmp/online.agent_telepath

agent_path=$dir_root/tmp/agent.batch.telepath
out_pred_path=$dir_root/tmp/predict.csv
env_id=SimpleGym_Sequential_Short-v0

# generate_replay
echo "python telepath.py generate_replay --config=$proj_spec --env_id=$env_id --replay_path=$replay_path --replay_spec=$replay_spec   --algo_spec=$algo_spec --agent_dest=$online_agent_path"
echo ""

#train
echo "python telepath.py train --config=$proj_spec --replay_path=$replay_path --replay_spec=$replay_spec   --algo_spec=$algo_spec --agent_path=$online_agent_path"
echo ""

#   todo: remove CartPole ident; read this data from the replay file

# predict
echo "python telepath.py predict --config=$proj_spec --in_features_path=$replay_path --in_spec=$replay_spec    --agent_path=$agent_path --out_pred_path=$out_pred_path"
echo ""


