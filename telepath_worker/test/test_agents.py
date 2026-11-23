import telepath_worker.ext_replay
from stable_baselines3.dqn.batch_dqn import BatchDQN
from stable_baselines3.dqn.dqn import DQN, DQNPolicy
from stable_baselines3.dqn import MlpPolicy

import gym

def _eval_agent(env, model, steps=100):
    obs = env.reset()
    for i in range(0,10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        #env.render()
        if done:
            obs = env.reset()


def test_dqn():
    env = gym.make('CartPole-v0')

    model = DQN(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=10, log_interval=1000)
    #model.save("dqn_pendulum")
    #del model  # remove to demonstrate saving and loading
    #model = DQN.load("dqn_pendulum")

    _eval_agent(env,model)


def test_batchdqn():
    env = gym.make('CartPole-v0')

    model = BatchDQN(MlpPolicy, env, verbose=1)

    # setup a buffer

    #model.learn_batch(total_timesteps=100, log_interval=1000)
    #model.save("dqn_pendulum")
    #del model  # remove to demonstrate saving and loading
    #model = DQN.load("dqn_pendulum")
    _eval_agent(env,model)


def test_batchdqn_saveload():
    env = gym.make('CartPole-v0')

    model = BatchDQN(MlpPolicy, env, verbose=1)

    # setup a buffer

    path = "batchdqn_agent.tel"
    # same some specs too?

    #model.learn_batch(total_timesteps=100, log_interval=1000)
    model.save(path)
    del model  # remove to demonstrate saving and loading
    model = BatchDQN.load(path)
    _eval_agent(env,model)



if __name__=="__main__":
    test_dqn()
    test_batchdqn()
