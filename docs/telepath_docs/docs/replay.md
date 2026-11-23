# Replay files

Replay files are the logs used for training.

## Replay CSV

Columns include:

- observations and next_observations
    - both usually involve multiple columns
- action taken
- reward observed
- done indicator (end of episode)

See Replay specs in [spec files](specs.md).

## Prediction input CSV
- should be in the same form as the training data
    - observation columns should be the same
    - sampleid or trajectoryid + stepid
    - without action, reward, done, next_observation columns

- can re-use the same replay spec yaml


## Prediction output CSV
- CSV file
- actions according to the agent
- sampleid to link rows back to prediction input
