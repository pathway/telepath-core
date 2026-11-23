from telepath_worker import evaluator
import gym
from stable_baselines3.dqn.batch_dqn import BatchDQN
from stable_baselines3.dqn.dqn import DQN, DQNPolicy
from stable_baselines3.dqn import MlpPolicy


def test_eval():
    env = gym.make('CartPole-v0')

    # better to have a random agent here
    model = DQN(MlpPolicy, env, verbose=1)
    model.agentid=1

    res = evaluator.evaluate_eps(env, 'online_agent', model, eval_eps=2);

    print("agent before learn", res)


if __name__=="__main__":
    test_eval()

