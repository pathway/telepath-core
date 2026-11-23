# Spec Files

Detailed settings are stored in yaml files.  Telepath calls these "spec files".

This keeps command lines simple, and makes it easy to store settings used for a specific run, by archiving them.


## Workflow

You will want a full set of spec files for your project, so start by copying and customizing template spec files.

As you iterate, you would typically iterate on the Algo, Replay, and Transform specs, as you evolve your algorithm settings, your observations and rewards.  

Checkpointing the spec files allows you to track the evolution of your agent in a repeatable way.


## Project spec yaml

- general settings for this project
- includes project identity
- usually do not change this while iterating on a project

```
---
  telepath_spec: 2          # do not change: format version
  spec_type: settings       # do not change: for validation
  content:
    tblog_folder: "s3://example-telepath-bucket/tblogs/"   # S3 or filesystem path for tensorboard logs
    label: "default"                           # short project label slug
    eval_eps: 3
```


## Algo spec yaml

- specifies parameters for RL training algorithm

```
---
  telepath_spec: 2
  spec_type: algo
  content:
    agent_learning_rate: 0.001
    agent_batch_size: 32
    total_samples: 500
    train_steps_per_agent: 500
    agent_target_update_interval: 10
    agent_learning_starts: 10
    env_gamma: 0.99
``` 

## Replay Spec yaml
- describes a Replay CSV file
- which columns correspond to:
     - obs_cols: observations
     - next_observations
     - reward
     - action
     - tid: trajectory id aka account id
     - step: sequential step in trajectory
     - sid: sample id
     - done: episode complete flag

```
---
  telepath_spec: 2
  spec_type: replay
  sample_strategy: rand
  chunksize: 10000
  n_actions: 2
  fieldmap:
      sid: sid
      tid: tid
      step: step
      done: done
      action: action
      reward: reward
      obs_cols:
        - obs0
        - obs1
        - obs2
        - obs3
      next_obs_cols:
        - nx_obs0
        - nx_obs1
        - nx_obs2
        - nx_obs3
```

## Prep spec

- used by "prep" command
- describes data preparation steps and transformations to be applied to replay buffer
- for details see [Prep scripts](prep.md)
