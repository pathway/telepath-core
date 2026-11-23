- want nice interface for managing runs like they were just actions; results as if they were rewards
- want nice interface for managing sets of agents, sets of envs, as if they were 1?
- want abstraction between remote and local
- want to manage complex pipelines (like env design with 2 opposed agents) with simple command
    - "Setting" as a first class citizen
    - Setting: SingleAgent/MultiAgent NoModel/FixedModel/LearnedModel SingleEnv/PoolEnv
    - MultiAgentSetting:
    - MultiAgentLearningEnvSetting

# second order systems (in general)

- automation: specify a bunch of envs and agent types.  AI manages run to characterize agents and envs over time.
    - Intent: I want information on performance of agents/envs
- experiments: understanding effect of settings / variations / ablations / comparisons
    - Intent: I want information on settings/comparisons
- hyperparameter tuning: finding best performing settings
    - Intent: I want information on settings/comparisons, and then I want to use that to maximize perf
- population base training; training pool of agents together
- multiagent: envs that admit multiple agents
   - game theory
   - adversarial training
   - hetrogenous games and agent types
- env-focussed
   - env evolution
   - env experiments
- agent+env focussed
   - michael dennis env design thingy
- learned worldmodels
- planning??


## example sessions

