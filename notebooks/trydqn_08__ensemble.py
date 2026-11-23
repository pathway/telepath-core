#import vaex
import gym
import random
from stable_baselines3 import DQN, BatchDQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys


'''
trydqn_08
- ensemble agent with modes

Next steps
- bootstrap dqn noise on ensemble members
    - need to figure out how this works
- distributional RL
    - replace dqn with qr/iqn
- replay exploration
    - UMAP embeddings
'''

class ReplayReturn():
    pass

class PdFrameBuffer():
    pos = 0
    buffer_size = 10000
    full = False

    df = pd.DataFrame()

    def add(self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray) -> None:
        # Copy to avoid modification by reference
        row = {'obs': np.array(obs).copy(),'next_obs': np.array(next_obs).copy(),
        'action': np.array(action).copy(),
        'reward' : np.array(reward).copy(),
        'done': np.array(done).copy()
        }
        self.df = self.df.append( row, ignore_index=True )
        #print(self.df)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0    # wrap around

    def sample(self, batch_size, env=None):
        # TODO: Simplify to speed this up
        """
        :param batch_size: (int) Number of element to sample
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: (Union[RolloutBufferSamples, ReplayBufferSamples])
        """
        upper_bound = len(self.df.index)
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds, env=None):
        samp = self.df.iloc[batch_inds,:]

        ret = ReplayReturn()
        npa = np.array( samp.next_obs.tolist() )
        ret.next_observations = th.tensor( npa )

        npa = np.array( samp.reward.tolist() )
        ret.rewards = th.tensor( npa )

        npa = np.array( samp.done.tolist() )
        ret.dones = th.tensor( npa )

        npa = np.array( samp.obs.tolist() )
        ret.observations = th.tensor( npa )

        npa = np.array( samp.action.tolist() )
        ret.actions = th.tensor( npa )

        return ret




class EnsembleAgent(object):
    '''
    Agent implements ensemble-mean and Thompson sampling methods
    '''
    members = []
    env = None

    def __init__(self, env):
        self.env = env
        pass

    def predict(self, obs, deterministic=True):
        default_mode_ens = 'mean'
        default_mode_selector = 'max'
        return self.predict_mode(obs, deterministic, mode_ens=default_mode_ens, mode_selector=default_mode_selector)

    def _get_ensemble_output(self, obs ):
        ens_action = []
        ens_q_list = []
        ens_state = []
        for idx, m in enumerate(self.members):
            obs = obs.reshape((-1,) + self.env.observation_space.shape)
            obs = th.as_tensor(obs) #.to(self.device)
            with th.no_grad():
                action, q_values, state = m.predict_raw(obs)
                ens_action.append(action)
                ens_q_list.append(q_values)
                ens_state.append(state)
        ens_q = th.stack(ens_q_list)
        return ens_q


    def predict_mode(self, obs, deterministic=True, mode_selector='max',mode_ens='mean' ):
        """
        deterministic=True: select action with "best" Q
        mode then determins how the "best" Q is determined.

        deterministic=False: thompson sampling
        """
        action_count = self.env.action_space.n
        obs_count = obs.shape[0]

        ens_q = self._get_ensemble_output(obs)

        if deterministic:
            if mode_ens=='median':
                ret = ens_q.median(axis=0)[0]
            elif mode_ens=='mean':
                ret = ens_q.mean(axis=0)[0]
            elif mode_ens=='max':
                ret = ens_q.max(axis=0)[0]
            elif mode_ens=='min':
                ret = ens_q.min(axis=0)[0]
            elif mode_ens=='std':
                ret = ens_q.std(axis=0)
            else:
                ret = ens_q.mean(axis=0)[0]

            if mode_selector =='max':
                ret2 = ret.argmax() # default to mean
            elif mode_selector =='min':
                ret2 = ret.argmin() # default to mean

            return ret2.cpu().reshape(-1).numpy(), None
        else:
            if mode == 'upper_thompson':
                #
                # Thompson sampling: for each action, select a single random Q, compare
                #qsa = [ random.choice( qmap[ai] ) for ai in range(0, action_count) ]
                #max_a = np.argmax( np.array(qsa))
                max_a = np.random.randint(0, action_count, size=obs_count)
                return max_a, None    # do not assume single observation

            elif mode=='uniform':
                max_a = np.random.randint(0, action_count, size=obs_count)
                return max_a, None    # do not assume single observation

            else:  # if mode=='thompson':
                # Thompson sampling: for each action, select a single random Q, compare
                #qsa = [ random.choice( qmap[ai] ) for ai in range(0, action_count) ]
                #max_a = np.argmax( np.array(qsa))
                max_a = np.random.randint(0, action_count, size=obs_count)
                return max_a, None    # do not assume single observation




'''
class VxFrameBuffer():
    pos = 0
    buffer_size = 10000
    full = False

    df = vx.Frame()

    def sample(self, batch_size, env=None):
        # TODO: Simplify to speed this up
        """
        :param batch_size: (int) Number of element to sample
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: (Union[RolloutBufferSamples, ReplayBufferSamples])
        """
        upper_bound = len(self.df.index)
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds, env=None):
        samp = self.df.iloc[batch_inds,:]

        ret = ReplayReturn()
        npa = np.array( samp.next_obs.tolist() )
        ret.next_observations = th.tensor( npa )

        npa = np.array( samp.reward.tolist() )
        ret.rewards = th.tensor( npa )

        npa = np.array( samp.done.tolist() )
        ret.dones = th.tensor( npa )

        npa = np.array( samp.obs.tolist() )
        ret.observations = th.tensor( npa )

        npa = np.array( samp.action.tolist() )
        ret.actions = th.tensor( npa )

        return ret
'''



def evaluate_eps(env, agent, num_eps=10, deterministic = True, mode_ens='mean', mode_selector='max'):
    # return mean episode reward by sampling some rollouts
    episode_rewards = [0.0]
    obs = env.reset()
    eps_count = 0
    needs_last_reward = False
    while eps_count < num_eps:
        if type(agent)==EnsembleAgent:
            action, _states = agent.predict_mode(obs,deterministic=deterministic, mode_ens=mode_ens, mode_selector=mode_selector)
        else:
            action, _states = agent.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)

        # Stats
        if needs_last_reward:
            episode_rewards.append(0.0)
            needs_last_reward = False

        episode_rewards[-1] += reward
        if done:
            obs = env.reset()
            needs_last_reward = True
            eps_count+=1

    a = episode_rewards
    decim = 3
    return { "mean" : round(np.mean(a), decim),

             "std" : round( np.std(a),decim ),
             "var" : round( np.var(a),decim ),

             "min" : round( np.min(a),decim ),
             "max" : round( np.max(a),decim ),
             "n": len(a),
             "q_05": round(np.quantile(a, 0.05), decim),

             "q_25": round(np.quantile(a,0.25), decim),
             "q_75": round(np.quantile(a, 0.75), decim),

             "q_95": round(np.quantile(a, 0.95), decim),

             #"list": episode_rewards
             }



def collect_rollouts( agent, env, replay, steps=1000, render=False, deterministic=True ):
    """
    Fill the agent's replay buffer with steps from agent + env
    """
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

import gym_stochastic.envs.simplebandits

'''
from gym.envs.registration import register

from gym_bandits.envs.bandits_env import MultiarmedBanditsEnv
class MultiarmedBanditFixed(MultiarmedBanditsEnv):
    """10 armed bandit with random probabilities assigned to payouts"""
    def __init__(self, bandits=10):
        p_dist = np.random.uniform(size=bandits)
        r_dist = np.full(bandits, 1)
        MultiarmedBanditsEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)

register(
    id='MultiarmedBanditFixed-v0',
    entry_point=':MultiarmedBanditFixed',
    # entry_point='gym_bandits.envs:MultiarmedBanditFixed',
)
'''
env_selector='bandit'
env_selector='cartpole'
env_selector='lunarlander'

import wandb
wandb.init(sync_tensorboard=True,project="ensemble3-"+env_selector)


def get_batch_agent_fn(env_selector, member_index):
    batch_agent_fn = 'batch_agent.' + env_selector + '.' + str(member_index) + '.telepath'
    return batch_agent_fn


if env_selector=='cartpole':
    agent_learning_rate=4e-3
    agent_batch_size=32
    agent_target_update_interval=100
    agent_learning_starts=100
    ensemble_size = 10
    env_gamma=0.99
    env_label = 'CartPole-v1';
    total_samples = 30000
    train_steps_per_agent = 800

elif env_selector=='bandit':
    agent_learning_rate=1e-3
    agent_batch_size=256
    agent_target_update_interval=100
    agent_learning_starts=100
    ensemble_size = 2
    env_label = 'MultiarmedBanditFixed-v0';
    env_gamma=0.0
    total_samples = 10000
    train_steps_per_agent = 1000

if env_selector=='lunarlander':
    agent_learning_rate=4e-3
    agent_batch_size=32
    agent_target_update_interval=100
    agent_learning_starts=100
    ensemble_size = 5
    env_gamma=0.99
    env_label = 'LunarLander-v2';
    total_samples = 30000
    train_steps_per_agent = 3000


#env = gym.make(env_label)
env = DummyVecEnv([lambda: gym.make( env_label )])

online_agent_path = "online_agent."+env_label+".bin"
online_agent_replay_path = "online_agent_replay"
full_replay = PdFrameBuffer()
full_replay.buffer_size = 50000

#mode='generate_replay'
#mode='train_offline'
mode='eval_offline'

if mode=='generate_replay':
    # --- try loading
    load_ok = False
    try:
        print("--- main: try loading online agent")
        online_agent = DQN.load(online_agent_path)
        print("loading native online agent replay")
        online_agent.load_replay_buffer(online_agent_replay_path+"/replay_buffer.pkl")

        print("loading full framebuffer replay")
        full_replay.df = pd.to_pickle('full_replay.pkl.zip', ) # this requires "pip install tables" ,
            # and for whatever reason, tables causes crash!

        load_ok = True
        print("LOADed ok")
    except:
        print("FAILED LOADING AGENT+BUFFER")
        pass

    if not load_ok:
        print("--- main: create agent and buffer")
        #online_agent = DQN('MlpPolicy', env, learning_starts=1000, verbose=1, learning_rate=0.001)
        online_agent = DQN('MlpPolicy', env, gamma=env_gamma, target_update_interval=agent_target_update_interval,
                learning_starts=100, verbose=1, learning_rate=agent_learning_rate,
                batch_size=agent_batch_size, tensorboard_log="/tmp/teletubby")

        res=evaluate_eps( env, online_agent,  ); print("online_agent before learn",res)
        print("online_agent.replay_buffer.pos",online_agent.replay_buffer.pos)

        for steps in [ total_samples,]: #20000]:
        #for steps in [20, 200]:
            online_agent.learn( log_interval=100, total_timesteps=steps, )
            print("online_agent.replay_buffer.pos",online_agent.replay_buffer.pos)
            res=evaluate_eps( env, online_agent,  ); print("online_agent after learn",res)

        for i in range(0,5):
            res = evaluate_eps(env, online_agent, ); print("online_agent after learn", res)

        #sys.exit()
        online_agent.save(online_agent_path)
        online_agent.save_replay_buffer(online_agent_replay_path)   # native sbl3 buffer -- needed?

        # convert replay to pandas and save as HD5
        # slow as balls
        print("transfering FULL replay...")
        onr = online_agent.replay_buffer
        for i in range(0,onr.pos):
            batch_inds = [i]
            full_replay.add(
               onr.observations[batch_inds, 0, :][0],
                onr.next_observations[batch_inds, 0, :][0],
                onr.actions[batch_inds, 0, :][0],
                onr.rewards[batch_inds][0],
                onr.dones[batch_inds][0],
            )
            if i % 500 ==0: print(i)

        #full_replay.df.to_feather('full_replay.fth',)
        print("--- main: save replay pickle ")
        full_replay.df.to_pickle('full_replay.'+env_selector+'.pkl.zip',)



if mode=='train_offline':
    print("--- main: load replay pickle ")
    full_replay.df = pd.read_pickle("full_replay."+env_selector+'.pkl.zip')

    print("--- main: create ensemble, train all members of ensemble ")

    common_replay = PdFrameBuffer() # install new replay buffer

    ea = EnsembleAgent(env)

    for es in range(0,ensemble_size):
        batch_agent = BatchDQN('MlpPolicy', env, verbose=1, learning_rate=agent_learning_rate, gamma=env_gamma,
                    batch_size=agent_batch_size, buffer_size=10, target_update_interval=agent_target_update_interval,
                    learning_starts=agent_learning_starts, device='cuda:0', tensorboard_log="/tmp/teletubby")
        del batch_agent.replay_buffer
        batch_agent.replay_buffer = common_replay
        res=evaluate_eps( env, batch_agent,  ); print("batch_agent before learn", res)

        print(full_replay.df)
        print("transfering replay...")

        batch_agent.replay_buffer.df = full_replay.df.copy() #.sample(total_samples)

        print("transfering replay...done")

        #batch_agent.replay_buffer.copy_from( online_agent.replay_buffer )
        print("batch_agent.replay_buffer.pos",batch_agent.replay_buffer.pos)

        #collect_rollouts( batch_agent, env, batch_agent.replay_buffer, steps=1000 )

        trainsteps=0
        trainsteps_add = train_steps_per_agent

        for i in range(0,1):
            batch_agent.learn_batch(total_timesteps=trainsteps_add,)
            print("batch_agent.replay_buffer.pos", batch_agent.replay_buffer.pos)
            res = evaluate_eps(env, batch_agent, );
            print("batch_agent after learn", res)
            trainsteps += trainsteps_add
            print("total trainsteps", trainsteps, res)

        batch_agent_fn = get_batch_agent_fn( env_selector, es )
        batch_agent.save( batch_agent_fn,)
        ea.members.append( batch_agent )

if mode=="eval_offline":
    print("--- main: load replay pickle ")
    full_replay.df = pd.read_pickle("full_replay."+env_selector+'.pkl.zip')


    print("--- main: load ensemble")
    ea = EnsembleAgent(env)
    for es in range(0,ensemble_size):
        batch_agent_fn = get_batch_agent_fn( env_selector, es )
        batch_agent = BatchDQN.load(batch_agent_fn)
        ea.members.append( batch_agent )



    print("--- main: try out some preds")

    #np_obs=np.expand_dims( np.array([-0.0041023744, -0.20529355, -0.017539551, 0.3]), axis=0 )
    obs=np.array([ [-0.0041023744, -0.20529355, -0.017539551, 0.3],
                   [+0.0041023744, 0.20529355, 0.017539551, 0.2],
                   [+0.0041023744, -0.20529355, 0.017539551, 0.1],] )


    eval_obs_subsample = full_replay.sample(1000, env=None).observations

    ensemble_q = ea._get_ensemble_output(eval_obs_subsample)
    print(ensemble_q)
    member_mean_deltas = (ensemble_q - ensemble_q.mean(axis=0))

    member_mean_deltas_neg = member_mean_deltas.clone()
    member_mean_deltas_neg[ member_mean_deltas_neg<0]=0
    member_mean_deltas_neg_std_score = member_mean_deltas_neg.std(axis=1).std(axis=1).cpu().numpy()

    member_mean_deltas_pos = member_mean_deltas.clone()
    member_mean_deltas_pos[ member_mean_deltas_pos>0]=0
    member_mean_deltas_pos_std_score = member_mean_deltas_pos.std(axis=1).std(axis=1).cpu().numpy()

    member_std_score = ( ensemble_q  - ensemble_q.mean(axis=0) ).std(axis=1).std(axis=1).cpu().numpy()

    member_obs_std_over_actions = ( ensemble_q  - ensemble_q.mean(axis=0) ).std(axis=2)
    member_action_std_over_obs  = ( ensemble_q  - ensemble_q.mean(axis=0) ).std(axis=1)

    member_std_score    # evaluate each member by how far its predictions are from mean
    # consider dropping members who are furthest from mean

    #obs=np.array([ [-0.0041023744, -0.20529355, -0.017539551, 0.3]] )

    import time

    res = ea.members[0].predict(obs, deterministic=False)

    try_pred_obs=0
    if try_pred_obs:
        for i in range(0,2):
            start = time.time()
            res = ea.predict( obs, deterministic=False )
            #print(res)
            end = time.time()
            print("det ms", (end - start)*1000.0)


        '''
        2ms for single pred or for 100 obs.
        Doesnt seem to matter which mode.
        '''

        for i in range(0,5):
            start = time.time()
            ensemble_mode = 'median'
            action, _states = ea.predict_mode(obs, deterministic=True, mode_selector='max', mode_ens=ensemble_mode)
            #print(res)
            end = time.time()
            print("det ms", (end - start)*1000.0)



        start = time.time()
        res = ea.predict( obs, deterministic=False )
        print(res)
        end = time.time()
        print("NOT det ms", (end - start)*1000.0)

    # CPU:
    # ensemble size  1: 0.3ms
    # ensemble size  3: 0.6ms
    # ensemble size  5: 1.0ms
    # ensemble size 10: 1.9ms
    # ensemble size 20: 3.5ms
    #
    # CUDA:
    # ensemble size  5: 1.7ms


    print("--- main: evaluate ensemble")


    df=pd.DataFrame()

    print(" + mode: thompson",)
    for i in range(0, 5):
        res = evaluate_eps(env, ea, deterministic=False );
        res.update({'mode': 'thompson'})
        df = df.append(res, ignore_index=True)
        print("thompson agent", res)


    print("--- main: evaluate members")
    for idx,ag in enumerate( ea.members):
        print(" member ",str(idx))
        for i in range(0,5):
            res = evaluate_eps(env, ag, )
            res.update({ 'mode': 'm_'+str(idx)  })
            df = df.append( res,ignore_index=True )
            print("member agent", res)


    for mode_selector,mode_ens in [('min','min'),('min','mean'),('min','std'),('max','std'),('max','min'),('max','max'),('max','mean',),('max','median'),]:
        print(" + mode: ", mode_selector,mode_ens)
        for i in range(0,10):
            res = evaluate_eps(env, ea, mode_ens=mode_ens, mode_selector=mode_selector);
            res.update({'mode_ens': mode_ens, 'mode_selector': mode_selector,'mode': mode_selector+'_'+mode_ens  })
            df = df.append( res,ignore_index=True )
            print("ensemble agent", res)
    print(df)
    df.to_csv('ensemble_agent_perf.'+env_selector+'.csv')
    exit()

    action, info = batch_agent.predict( obs )
    print("action",action,info)
    batch_agent.replay_buffer.df['q']=0


    # --- fill in q/v/adv/pi to dataframe for analysis

    q_list = []
    val_list = []
    adv_list = []
    pi_list = []
    for i in range(0,len(batch_agent.replay_buffer.df.index)):
        row = batch_agent.replay_buffer.df.iloc[i,:]
        print(batch_agent.replay_buffer.df.iloc[i,:])
        #print(row.obs)
        #action, info = batch_agent.predict( row.obs )
        #print("action",action,info)

        obs = row.obs.reshape((-1,) + env.observation_space.shape)
        obs = th.as_tensor(obs) #.to(self.device)
        with th.no_grad():
            qs, action = batch_agent.policy.q_net._predict_raw(obs,)

        #qs = batch_agent.policy._predict( th.Tensor(row.obs), deterministic=True )
        #print("qs action",qs,action)
        npa = np.array( qs.numpy()[0] )
        #batch_agent.replay_buffer.df.q[i] = npa
        q_list += [ npa, ]
        val_list += [ max(npa) ]
        adv_list += [ npa-max(npa) ]
        pi_list += [ action.numpy()[0] ]

    batch_agent.replay_buffer.df['q'] = q_list
    batch_agent.replay_buffer.df['v'] = val_list
    batch_agent.replay_buffer.df['a'] = adv_list
    batch_agent.replay_buffer.df['pi'] = pi_list
    print(batch_agent.replay_buffer.df[['q','v','a','pi']])

    env.close()
