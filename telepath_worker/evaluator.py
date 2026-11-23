import numpy as np
#import telepath.registry

MLFLOW=0
if MLFLOW:
    import mlflow

# TODO: note this evaluator requires a number of episodes.
# how do we deal with non-episodic?

def evaluate_eps(env, agent_label, telepath_agent, eval_eps=10, deterministic = True, return_raw_scores=False, mlflow_folder=None):
    if MLFLOW:
        #mlflow.set_experiment('PolicyEval')
        mlflow.set_experiment('PolicyEval')
        print(mlflow_folder)
        mlflow.set_tracking_uri(mlflow_folder)
        #mlflow.set_registry_uri(mlflow_folder)
    #with mlflow.start_run(run_name=agent_label, ) as run:
    if 1:
            # return mean episode reward by sampling some rollouts
        episode_rewards = [0.0]
        obs = env.reset()
        eps_count = 0
        needs_last_reward = False
        while eps_count < eval_eps:
            action, _states = telepath_agent.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)

            # Stats
            if needs_last_reward:
                episode_rewards.append(0.0)
                needs_last_reward = False

            episode_rewards[-1] += reward.item()
            if done:
                obs = env.reset()
                needs_last_reward = True
                eps_count+=1

        a = episode_rewards
        decim = 3
        ret= { "mean" : round(np.mean(a), decim),

            "std" : round( np.std(a),decim ),
            "var" : round( np.var(a),decim ),

            "min" : round( np.min(a),decim ),
            "max" : round( np.max(a),decim ),
            "n": len(a),
            "q_05": round(np.quantile(a, 0.05), decim),
            "q_25": round(np.quantile(a,0.25), decim),

            "q_50": round(np.quantile(a,0.5), decim),

            "q_75": round(np.quantile(a, 0.75), decim),
            "q_95": round(np.quantile(a, 0.95), decim),

            #"list": episode_rewards
            }
        if return_raw_scores:
            ret['scores'] = a

        if MLFLOW:
            mlflow.log_param('env',env.envs[0].spec.id)
            mlflow.log_param('agentid',telepath_agent.getid() )
            mlflow.log_metrics(ret)
            mlflow.end_run()

        ret['agentid']=telepath_agent.getid()

        return ret


