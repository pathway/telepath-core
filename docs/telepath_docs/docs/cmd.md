# Telepath Command line

Think of commands like verbs.  Typically the overall Telepath workflow will involve a number of different command lines in a sequence.  Detailed settings are stored in [spec files](specs.md) to allow for testing and audit trails.

__Core commands__

- __prep__: prepare training data
- __train__: learn agent from training data
- __predict__: use agent to predict actions from prediction features

__Analysis commands__

- __replay_stats__: get stats on a replay file 
- __explain__: posthoc explainations 
- __compare__: compare and explain differences

__Utility commands__

- __generate_replay__: synthesize training data for testing
- __tb__: tensorboard visualization
- __procx__: multiprocessing

## Workflow

To quickly get started, generate a sample workflow with [integration test script generator](integ.md).


## Prep
- read in raw training data "in_replay"
- transform and write replay file "out_replay"
- for details see [Prep scripts](prep.md)

```
telepath prep \
     --project_spec=spec_project.yaml \
     --in_replay=replay_rawtrain.csv \
     --out_replay=replay.csv \
     --prep_spec=spec_replay_prep.yaml
```

Options

-  `--include_obs` : Include observations in the output file.  This makes it a bit easier to relate actions to the observations.

- `--include_q` : Include Q values in the output file.  This helps debug the agent and produces summaries and charts like this: 

<br>


## Train
- read in replay data from _replay_path_
- train and write agent to _agent_path_

```
./telepath train \
     --project_spec=telepath_settings.yaml \
     --replay=replay.csv \
     --replay_spec=replay_spec.yaml   \
     --algo_spec=tiny_spec_algo.yaml \
     --agent_path=_batch.agent_telepath
```

Notes

- `agent_path` : if this path contains the string AGENTID, it is replaced with the unique ID of this agent ("A" followed by random sequence). This allows you to generate multiple agents having [unique agent names](uniq.md) with a single command, which can be useful when running [multiple processes](procx.md).

<br>

## Predict
- read in features and existing agent
- generate predicted actions
- output in `out_pred_path` has column named `batch_agent_action` holding result for each row

```
./telepath predict \
     --project_spec=telepath_settings.yaml \
     --in_features_path=replay.csv \
     --in_spec=replay_spec.yaml    \
     --agent_path=_batch.agent_telepath \
     --out_pred_path=_predict.csv
```

Notes

- `agent_path` : predict with mutiple agents by providing a comma-separated list of agent paths, in which case actions will be output in columns `agent0_action`, `agent1_action` etc.
- `include_obs` : include observations in the output file
- `include_next` : include next_observations in the output file
- `include_r` : include reward in the output file

<br>



## Replay stats

Show summary stats for replay data.  Useful to check assumptions about the data.

```
./telepath replay_stats --replay=s3://bucket/replay.csv
```
In particular, check the isnull, count, min/max columns, and column names are as expected.

Options

-  `--col=<colname>` (optional) : if set, show histogram for a single column

<br>

## Explain

Explain a column using observation columns.

Useful for:

- explaining actions of agent
- explaining Q values of agent
- explaining next_obs column in terms of obs columns 

```
python ./telepath.py explain \
  --project_spec=spec_project.yaml \
  --replay=predict.csv \
  --replay_spec=spec_replay.yaml \
  --explain_col=obs1 \
  --max_depth=3 \
  --type=class
```

Explanations are using telescoped decision trees for classification or regression.

Options

- `--explain_col=<colname>` : which column to explain
- `--max_depth` : show decision trees of at most this depth
- `--valtype` : either `continous` for continuous explain_col, or `discrete` for discrete

<br>


## Compare

Compare two columns using observation columns, and explain the difference.  Useful for:

- comparing actions of 2 agents
- comparing Q values for 2 agents
- comparing obs and next_obs

```
python ./telepath.py compare \
  --project_spec=spec_project.yaml \
  --replay=predict.csv \
  --replay_spec=spec_replay.yaml \
  --col1=agent0_action \
  --col2=agent1_action \
  --max_depth=3 \
  --valtype=class
```

Options

- `--col1=<colname>`, `--col2=..` : which columns to compare
- `--max_depth` : show decision trees of at most this depth
- `--valtype` : either `continous` for continuous (col1,col2), or `discrete` for discrete.  Both col1 and col2 should be of the same type.

<br>


## Generate replay data
- use a gym to produce a synthetic replay file
- useful for testing

```
./telepath generate_replay \
     --project_spec=telepath_settings.yaml \
     --env_id=SimpleGym_Sequential_Short-v0 \
     --out_replay=replay_rawtrain.csv \
     --replay_spec=replay_spec.yaml   \
     --algo_spec=tiny_spec_algo.yaml \
     --agent_dest=_online.agent_telepath
```

<br>


## Tensorboard

Launch tensorboard.  See [Tensorboard](tb.md).

<br>

## Multiprocessing
See  [Multiprocessing with procx](procx.md)

<br>



