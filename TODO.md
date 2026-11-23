# mvp use cases

client can ...

#### recommended folder set
- which files are separate / together

#### validate, featurize their dataset
- P0 validate training dataset
    - P1 prep : identity, standardize, featurize
- P0 validate prediction dataset


#### accept an arbirary csv as environment spec
- create env spec directly from replay spec
- weakest

#### train policy on a dataset
- P0 run 

#### predict with policy on a dataset
- weakest

-----

# by component

## replay

- bug: tblogs written to _local dir_ s3://
- P0 write high perf file from csv
- P0 normalizing
    - find: self._normalize_obs

- P0 shuffling
- P0 generate next_obs + validate: better on vx
- P1 real tests for handling massive files
- replay designed to allow for:
    - P0 training Q-learning agents
    - P1 sequential traversal (vs random) as a first class thing
    - P2 training world models, inverse action models, other things
- P2 consider tests for effect of replay buffer randomization

