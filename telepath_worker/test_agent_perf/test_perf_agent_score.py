from telepath_worker import evaluator
from telepath_worker import telepath
import gym
from stable_baselines3.dqn.batch_dqn import BatchDQN
from stable_baselines3.dqn.dqn import DQN, DQNPolicy
from stable_baselines3.dqn import MlpPolicy
from telepath_worker import spec
from telepath_worker.telepath import get_vec_env_from_label
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    EveryNTimesteps,
    StopTrainingOnRewardThreshold,
)


def test_perf_agent_online_score_base():
    env_id = 'SimpleGym_Sequential_Short-v0'

    env = gym.make(env_id)

    # I need to specify args with
    algsp = spec.AlgoSpec()
    algsp.load('specs2/tiny_spec_algo.yaml')
    print(algsp.spec)
    algo_ = algsp.spec['content']

    settings_file = 'specs2/telepath_settings.yaml'
    setspec = spec.SettingsSpec()
    setspec.load(settings_file)

    set_ = setspec.spec['content']
    vec_env_sim = get_vec_env_from_label(env_id)

    online_agent = telepath._get_online_agent(algo_, set_, vec_env_sim)

    online_agent.learn(log_interval = 10,
        total_timesteps = algo_['total_samples'],
        eval_env = vec_env_sim,
        # callback=eval_callback,
        tb_log_name = 's_dqn_' + online_agent.agentid  )

    res = evaluator.evaluate_eps(vec_env_sim, 'online_agent', online_agent, eval_eps=10);
    print("agent after learn", res)

    assert res['mean']>5.0  # should be 6.0
    assert res['mean']<7.0  # impossible



def test_perf_agent_online_score_cartpole():
    env_id = 'CartPole-v0'

    # I need to specify args with
    algsp = spec.AlgoSpec()
    algsp.load('specs2/tiny_spec_algo_cartpole_perf.yaml')
    print(algsp.spec)
    algo_ = algsp.spec['content']

    settings_file = 'specs2/telepath_settings.yaml'
    setspec = spec.SettingsSpec()
    setspec.load(settings_file)

    set_ = setspec.spec['content']
    vec_env_sim = get_vec_env_from_label(env_id)

    online_agent = telepath._get_online_agent(algo_, set_, vec_env_sim)

    eval_callback = EvalCallback(
        vec_env_sim,
        # callback_on_new_best=callback_on_best,
        # best_model_save_path='agent_best_save',
        log_path=set_['tblog_folder'],  # this is NOT s3 compatible, and Im not sure how to make it so
        eval_freq=1000,
        n_eval_episodes=1,
    )

    online_agent.learn(log_interval = 10,
        total_timesteps = algo_['total_samples'],
        eval_env = vec_env_sim,
        callback=eval_callback,
        tb_log_name = 's_dqn_' + online_agent.agentid  )

    res = evaluator.evaluate_eps(vec_env_sim, 'online_agent', online_agent, eval_eps=10);
    print("agent after learn", res)

    assert res['mean']>40.0  #
    assert res['mean']<202.0  # impossible



def test_perf_agent_online_score_lunar1():
    env_id = 'LunarLander-v2'

    # I need to specify args with
    algsp = spec.AlgoSpec()
    algsp.load('specs2/tiny_spec_algo_lunar1_perf.yaml')
    print(algsp.spec)
    algo_ = algsp.spec['content']

    settings_file = 'specs2/telepath_settings.yaml'
    setspec = spec.SettingsSpec()
    setspec.load(settings_file)

    set_ = setspec.spec['content']
    vec_env_sim = get_vec_env_from_label(env_id)

    online_agent = telepath._get_online_agent(algo_, set_, vec_env_sim)

    eval_callback = EvalCallback(
        vec_env_sim,
        # callback_on_new_best=callback_on_best,
        # best_model_save_path='agent_best_save',
        log_path=set_['tblog_folder'],  # this is NOT s3 compatible, and Im not sure how to make it so
        eval_freq=1000,
        n_eval_episodes=1,
    )

    online_agent.learn(log_interval = 10,
        total_timesteps = algo_['total_samples'],
        eval_env = vec_env_sim,
        callback=eval_callback,
        tb_log_name = 's_dqn_' + online_agent.agentid  )

    res = evaluator.evaluate_eps(vec_env_sim, 'online_agent', online_agent, eval_eps=10);
    print("agent after learn", res)

    assert res['mean']>-300.0  #
    assert res['mean']<500.0  # impossible




if __name__=="__main__":
    #test_perf_score_base()
    #test_perf_score_cartpole()
    test_perf_score_lunar1()
