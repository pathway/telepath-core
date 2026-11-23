# Telepath Policy Platform


## Overview

Telepath is a machine learning platform for the private cloud.  It encapsulates deep reinforcement learning, deep learning, and other machine learning methods.

It is designed to optimize a process to maximize some desired result, using your logs of past interactions. 

This is done by learning policies with deep reinforcement learning.


## Execution Environment + Model

Telepath is provided via a [docker](docker.md) image.  That means you can run telepath:

- on your private [Amazon Web Services](aws.md) cloud
    - via ECS
    - via AWS Batch
- anywhere docker runs

Training requires a GPU and nvidia CUDA libraries.

It currently runs in batch mode.

The interface is via [command line](cmd.md).

Settings are done via [yaml spec files](specs.md).

Input training data, input feature data, and output prediction data use [csv files](replay.md).


## Reinforcement learning approach

Telepath currently supports these settings:

- discrete action model-free offline deep reinforcement learning, and 
- discrete action model-free offline deep bandits (by setting gamma=0.0 to produce myopic policies.)

The main algorithm family is [DQN by Mnih et al](https://www.nature.com/articles/nature14236).

The DQN algorithm implementation is currently based on a modified fork of [stable baselines 3](https://stable-baselines3.readthedocs.io/en/master/).

Additional algorithms will be supported in future.
