# Telepath

**Offline reinforcement learning from logs of past decisions, as a command-line tool.**

You give Telepath a CSV log of what happened: the situation (observations), the action taken, the reward that followed, and the next situation. It trains a deep reinforcement learning policy from that log, with no live environment needed, and then predicts the best action for new situations.

Typical question it answers: *given what we did before and what resulted, which action should we take next?*

```
 raw log (CSV)                     spec files (YAML)
      │                                  │
      ▼                                  ▼
 ┌─────────┐   replay CSV   ┌─────────┐   agent file   ┌───────────┐
 │  prep   │ ─────────────▶ │  train  │ ─────────────▶ │  predict  │ ─▶ actions (+ Q values)
 └─────────┘                └─────────┘                └───────────┘
                                 │                          │
                                 ▼                          ▼
                       tensorboard logs        explain · compare · replay_stats
```

## What it does

- **Offline deep RL** for discrete actions: DQN-family agents trained from a replay file.
- **Bandit mode**: set `env_gamma: 0.0` for myopic policies that optimise the next reward only.
- **Command line plus YAML specs**: every setting of a run lives in a spec file you can archive, diff and rerun.
- **Data preparation scripts** (`prep`): sort, fill, shift, roll and derive columns in a short script, so the same steps apply to training data and to prediction data.
- **Analysis commands**: summarise a replay file, explain a policy, compare two agents, view training in TensorBoard.
- **Runs in Docker** on a machine with an NVIDIA GPU, including AWS ECS or AWS Batch. Files can be local paths or `s3://` URLs.
- **Parallel training** of several agents at once with `procx`, for ensembles, stability checks and hyperparameter search.
- **Test environments**: small built-in gyms for trying the whole workflow without real data.

## Commands

| Command | What it does |
|---|---|
| `prep` | Transform raw log data into a replay file using a prep script |
| `train` | Learn an agent from a replay file |
| `predict` | Choose actions for new observations with a trained agent (`--include_q`, `--include_obs`, `--explain`) |
| `replay_stats` | Summarise a replay file |
| `explain` | Post-hoc explanation of a policy |
| `compare` | Compare and explain the differences between agents |
| `generate_replay` | Synthesise data from a built-in gym, for testing |
| `tb` | TensorBoard visualisation |
| `procx` | Run several processes in parallel |

## Quick start

Telepath is built as a Docker image. Training needs a GPU and the [NVIDIA container runtime](https://github.com/NVIDIA/nvidia-docker).

```bash
docker build -t telepath .
docker container run --gpus all -it telepath nvidia-smi        # check GPU access
```

The container can write out a full example workflow for you. This one uses a built-in gym, so it needs no data of your own:

```bash
docker container run --gpus all -it telepath integ_gen \
  --spec_root=/tmp/test1/spec \
  --data_root=/tmp/test1/data \
  --env_id=SimpleGym_Sequential_Short-v0 \
  --exec_prefix="docker container run --gpus all -it telepath"
```

It prints the `generate_replay`, `prep`, `train` and `predict` command lines for that project, in order. Run them one after another.

A hand-written run looks like this:

```bash
telepath prep \
  --project_spec=spec_project.yaml \
  --in_replay=replay_rawtrain.csv \
  --out_replay=replay.csv \
  --prep_spec=spec_replay_prep.yaml

telepath train \
  --project_spec=spec_project.yaml \
  --replay=replay.csv \
  --replay_spec=spec_replay.yaml \
  --algo_spec=spec_algo.yaml \
  --agent_path=agent_AGENTID.telepath

telepath predict \
  --project_spec=spec_project.yaml \
  --in_features_path=features.csv \
  --in_spec=spec_replay.yaml \
  --agent_path=agent_A1b2c3.telepath \
  --out_pred_path=predictions.csv \
  --include_q
```

`AGENTID` in an agent path is replaced by a unique agent ID, so repeated or parallel runs never overwrite each other.

## Spec files

Three small YAML files describe a project.

```yaml
# spec_algo.yaml: how to train
telepath_spec: 2
spec_type: algo
content:
  agent_learning_rate: 0.001
  agent_batch_size: 32
  total_samples: 500
  train_steps_per_agent: 500
  agent_target_update_interval: 10
  agent_learning_starts: 10
  env_gamma: 0.99          # 0.0 gives a bandit (myopic) policy
```

```yaml
# spec_replay_prep.yaml: how to prepare raw data
telepath_spec: 2
spec_type: replay_prep
content:
  - prep_mode: all
    script:
      - sort user_id day
      - fillna0 recent_event_count day_of_week
```

The project spec holds the project label, evaluation settings and where TensorBoard logs go. The replay spec says which columns are observations, next observations, action, reward, trajectory id and step. See [docs/telepath_docs/docs/specs.md](docs/telepath_docs/docs/specs.md).

## Replay files

One row per step. Columns:

- observations, and next observations (usually several columns each)
- the action taken
- the reward observed
- a done flag marking the end of an episode
- a trajectory id and a step number, to order rows within a trajectory

Prediction input has the same layout without the action, reward, done and next-observation columns.

## Terms

Telepath uses reinforcement learning vocabulary:

| Machine learning | Telepath |
|---|---|
| input features | observations |
| training data | replay data, replay buffer |
| model | policy, agent |

## Running without Docker

```bash
conda env create -f environment.yaml          # Python 3.8; pinned versions of 2020 in requirements.txt
pip install git+https://github.com/pathway/stable-baselines3-telepath.git
git clone https://github.com/pathway/fqf-iqn-qrdqn.pytorch.git fqf && export PYTHONPATH=$PWD
pip install .
cd telepath_worker && ./telepath --help
```

Tests: `cd telepath_worker && pytest test/`. Some tests need data files that are not in this repository.

## Repository layout

| Path | Contents |
|---|---|
| `telepath_worker/` | The `telepath` command, agents, replay handling, specs, storage layer, tests |
| `telepath_gym/` | Small built-in environments for testing |
| `deepgym/` | Learned-model environments |
| `syscheck/` | A test container that checks GPU, container-registry and S3 access |
| `aws/`, `devops/`, `airflow/`, `docker/` | AWS helpers, release scripts, scheduling examples, container files |
| `docs/telepath_docs/` | User documentation (MkDocs): commands, specs, prep scripts, replay files, AWS, Docker |
| `docs/dev_docs/`, `design/` | Developer notes and design sketches |
| `workbench/`, `junk/` | Scratch experiments |

Build the documentation with `cd docs/telepath_docs && mkdocs serve`.

## Related repositories

- [stable-baselines3-telepath](https://github.com/pathway/stable-baselines3-telepath): a fork of Stable-Baselines3 that adds `BatchDQN`. Telepath's DQN agents are built on it.
- [fqf-iqn-qrdqn.pytorch](https://github.com/pathway/fqf-iqn-qrdqn.pytorch): FQF, IQN and QR-DQN in PyTorch, with offline QR-DQN additions.

## Status

Written in 2020 and 2021 (version 0.8.1.15). It trains discrete-action, model-free, offline agents in batch mode. The pinned dependency versions date from that period, so expect to update them. Prediction speed was measured at about one million predictions in 16 to 23 minutes on one GPU workstation; training time ranges from half an hour to days, depending on data and settings.

## License

MIT. Copyright (c) 2020-2021 Pathway Intelligence Inc. Contact: robin@pathwayi.com
