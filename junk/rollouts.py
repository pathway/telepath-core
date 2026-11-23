#import telepath.registry
MLFLOW=0
if MLFLOW:
    import mlflow

# not used?


def collect_rollouts( agent, agent_label, env, replay, steps=1000, render=False, deterministic=True ):
    """
    Fill the agent's replay buffer with steps from agent + env
    """
    if MLFLOW:
        #uri="s3:////mlflow/
        #mlflow.set_tracking_uri(uri)
        #mlflow.set_experiment('Rollouts')
        mlflow.set_experiment('Rollouts2')
    if 0:
        #with mlflow.start_run(run_name='rollout_'+agent_label, ) as run:

        if MLFLOW: mlflow.log_param('env',env.envs[0].spec.id)

        obs = env.reset()
        for i in range(steps):
            action, _states = agent.predict(obs, deterministic=deterministic)
            last_obs = obs
            obs, reward, done, info = env.step(action)

            replay.add(last_obs, obs, action, reward, done)
            #print(agent.replay_buffer.pos)
            if render:
                env.render()
            if done:
              obs = env.reset()

        if MLFLOW:
            mlflow.log_metric('steps',steps)
            mlflow.log_param('env',env.envs[0].spec.id)
            mlflow.end_run()

