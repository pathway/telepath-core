'''
mlflow experiment tracking on aws:

https://towardsdatascience.com/deploy-mlflow-with-docker-compose-8059f16b6039

https://medium.com/@josemrivera/running-mlflow-on-aws-to-track-your-machine-learning-experiments-e192830f996c

https://medium.com/@alexanderneshitov/how-to-run-an-mlflow-tracking-server-on-aws-ec2-d7afd0ac8008

https://github.com/flmu/mlflow-tracking-server
https://karimlahrichi.com/2020/03/06/basic-mlflow-setup-to-track-your-experiments-on-aws-ec2/

https://cosminsanda.com/posts/experiment-tracking-with-mlflow-inside-amazon-sagemaker/

Next steps
- replay exploration
    - UMAP embeddings
'''


import gym
import random
from stable_baselines3 import DQN, BatchDQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sys
import mlflow
import mlflow.pytorch
import boto3

import gym_stochastic.envs.simplebandits

from ensemble_agent import EnsembleAgent
from evaluator import evaluate_eps
from ext_replay import PdFrameBuffer
from rollouts import collect_rollouts

import storage_layer as storage

ENABLE_WAND=0

import yaml

conf=None
sett = {}
ea = None
full_replay = PdFrameBuffer()
full_replay.buffer_size = 50000


def settings():
    global conf
    #env = gym.make(env_label)
    sett['env'] = DummyVecEnv([lambda: gym.make( conf['env_label'] )])

    sett['online_agent_path'] = conf['folder'] + "agents/online_agent."+conf['env_label']+".bin"
    sett['online_agent_replay_path'] = "online_agent_replay/online_agent_replay"

    sett['replay_loc'] = conf['folder'] + "replay/full_replay."+env_selector+".pkl.zip"
    sett['mem_perf_fn'] = conf['folder'] + 'agent_perf/df_mem.' + env_selector + '.csv'
    sett['ens_perf_fn']='agent_perf/ensemble_agent_perf.' + env_selector + '.csv'



def get_batch_agent_fn(env_selector, member_index):
    batch_agent_fn = conf['folder'] + 'agents/batch_agent.' + env_selector + '.' + str(member_index) + '.telepath'
    return batch_agent_fn


def main_generate_replay():
    # --- try loading
    load_ok = False
    if 0:
        try:
            print("--- main: try loading online agent")
            online_agent = DQN.load(online_agent_path)
            print("loading native online agent replay")
            online_agent.load_replay_buffer(sett['replay_loc'])

            print("loading full framebuffer replay")
            full_replay.df = pd.to_pickle(sett['replay_loc']) # this requires "pip install tables" ,
                # and for whatever reason, tables causes crash!

            load_ok = True
            print("LOADed ok")
        except:
            print("FAILED LOADING AGENT+BUFFER")
            pass

    if not load_ok:
        print("--- main: create agent and buffer")
        #online_agent = DQN('MlpPolicy', env, learning_starts=1000, verbose=1, learning_rate=0.001)
        online_agent = DQN('MlpPolicy', sett['env'], gamma=conf['env_gamma'], target_update_interval=conf['agent_target_update_interval'],
                learning_starts=conf['agent_learning_starts'], verbose=1, learning_rate=conf['agent_learning_rate'],
                batch_size=conf['agent_batch_size'], tensorboard_log="/tmp/teletubby")

        res=evaluate_eps( sett['env'], 'online_agent', online_agent, eval_eps=conf['eval_eps']  ); print("online_agent before learn",res)
        print("online_agent.replay_buffer.pos",online_agent.replay_buffer.pos)

        for steps in [ conf['total_samples'],]: #20000]:
        #for steps in [20, 200]:
            online_agent.learn( log_interval=100, total_timesteps=steps, )
            print("online_agent.replay_buffer.pos",online_agent.replay_buffer.pos)
            res=evaluate_eps( sett['env'], 'online_agent', online_agent, eval_eps=conf['eval_eps']  ); print("online_agent after learn",res)

        for i in range(0,conf['eval_reps']):
            res = evaluate_eps(sett['env'], 'online_agent', online_agent, eval_eps=conf['eval_eps'] ); print("online_agent after learn", res)

        #sys.exit()
        #online_agent.save(online_agent_path)
        storage.save_agent_location( online_agent,sett['online_agent_path'] )
        online_agent.save_replay_buffer(sett['online_agent_replay_path'])   # native sbl3 buffer -- needed?

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
            #if i % 500 ==0: print(i)
        print("...done")

        #full_replay.df.to_feather('full_replay.fth',)
        print("--- main: save replay pickle ")
        storage.write_pickle_to_location( full_replay.df, sett['replay_loc'])





def main_train_offline():
    import wandb
    if ENABLE_WAND:
        wandb.init(sync_tensorboard=True, project="tel3-" + conf['label'])

    print("--- main: load replay pickle ")
    full_replay.df = storage.read_df_pickle_from_location(sett['replay_loc'])

    print("--- main: create ensemble, train all members of ensemble ")

    common_replay = PdFrameBuffer() # install new replay buffer

    ea = EnsembleAgent(sett['env'])

    for es in range(0,conf['ensemble_size']):
        batch_agent = BatchDQN('MlpPolicy', sett['env'], verbose=1, learning_rate=conf['agent_learning_rate'], gamma=conf['env_gamma'],
                    batch_size=conf['agent_batch_size'], buffer_size=10, target_update_interval=conf['agent_target_update_interval'],
                    learning_starts=conf['agent_learning_starts'], device='cuda:0', tensorboard_log="/tmp/teletubby")
        del batch_agent.replay_buffer
        batch_agent.replay_buffer = common_replay
        res=evaluate_eps( sett['env'], 'batch_agent', batch_agent, eval_eps=conf['eval_eps'] ); print("batch_agent before learn", res)

        #print(full_replay.df)
        print("transfering replay...")

        batch_agent.replay_buffer.df = full_replay.df.copy() #.sample(total_samples)

        print("transfering replay...done")

        #batch_agent.replay_buffer.copy_from( online_agent.replay_buffer )
        print("batch_agent.replay_buffer.pos",batch_agent.replay_buffer.pos)

        #collect_rollouts( batch_agent, env, batch_agent.replay_buffer, steps=1000 )

        trainsteps=0
        trainsteps_add = conf['train_steps_per_agent']

        for i in range(0,1):
            batch_agent.learn_batch(total_timesteps=trainsteps_add,)
            print("batch_agent.replay_buffer.pos", batch_agent.replay_buffer.pos)
            res = evaluate_eps(sett['env'], 'batch_agent',batch_agent, eval_eps=conf['eval_eps'] );
            print("batch_agent after learn", res)
            trainsteps += trainsteps_add
            print("total trainsteps", trainsteps, res)

        batch_agent_fn = get_batch_agent_fn( env_selector, es )
        # batch_agent.save( batch_agent_fn,)
        storage.save_agent_location( batch_agent, batch_agent_fn )
        ea.members.append( batch_agent )




def get_ensemble_agent():
    print("--- main: load ensemble")
    ea = EnsembleAgent(sett['env'])
    for es in range(0,conf['ensemble_size']):
        batch_agent_fn = get_batch_agent_fn( env_selector, es )
        #batch_agent = BatchDQN.load(batch_agent_fn)
        batch_agent = storage.load_agent_location( BatchDQN, batch_agent_fn)
        ea.members.append( batch_agent )
    return ea


def main_eval_offline_members(ea):

    print("--- main: evaluate ensemble")
    df=pd.DataFrame()

    print("--- main: evaluate members")
    for idx,ag in enumerate( ea.members):
        print(" member ",str(idx))
        for i in range(0,conf['eval_reps']):
            res = evaluate_eps(sett['env'], 'composite_member_'+str(i), ag, eval_eps=conf['eval_eps'] )
            res.update({ 'member': str(idx)  })
            df = df.append( res,ignore_index=True )
            print("member agent", res)
    return df


def main_eval_offline_ensemble(ea):
    df=pd.DataFrame()

    print(" + mode: thompson",)
    for i in range(0, conf['eval_reps']):
        res = evaluate_eps(sett['env'], 'composite_agent',ea, deterministic=False, eval_eps=conf['eval_eps'] );
        res.update({'mode': 'thompson'})
        df = df.append(res, ignore_index=True)
        print("thompson agent", res)

    low_modes = [('min','min'),('min','mean'),('min','std'),('max','std'),]
    all_modes = [('max','median'),('max','mean',),('max','min'),('max','max'),]
    modes = [('max','mean',),('max','median'),]

    for mode_selector,mode_ens in all_modes:
        print(" + mode: ", mode_selector,mode_ens)
        for i in range(0,conf['eval_reps']):

            ea.mode_selector = mode_selector
            ea.mode_ens = mode_ens

            res = evaluate_eps(sett['env'], 'composite_agent', ea, eval_eps=conf['eval_eps']);
            res.update({'mode_ens': mode_ens, 'mode_selector': mode_selector,'mode': mode_selector+'_'+mode_ens  })
            df = df.append( res,ignore_index=True )
            print("ensemble agent", res)
    print(df)

    return df





def call_eval_members():
    ea = get_ensemble_agent()
    ea_size = len(ea.members)
    df_mem = main_eval_offline_members(ea)
    storage.write_df_to_location(df_mem, sett['mem_perf_fn'])

def call_eval_ensemble():
    ea = get_ensemble_agent()
    df_mem = storage.read_df_from_location(sett['mem_perf_fn'])
    df_ens = main_eval_offline_ensemble(ea)
    storage.write_df_to_location( df_ens, sett['ens_perf_fn'])



def call_partial_ensemble():
    ea = get_ensemble_agent()
    df_mem = storage.read_df_from_location(perf_fn)
    # only include top half of x-quartile performers

    qq=df_mem['mean'].quantile(q=0.25)
    res = (df_mem['mean'] >= qq)
    int_res_min = [ int(r) for r in res]
    print('int_res_min',int_res_min)

    qq=df_mem['mean'].quantile(q=0.75)
    res = (df_mem['mean'] <= qq)
    int_res_max = [ int(r) for r in res]
    print('int_res_max',int_res_max)

    ea.member_selector = [ int_res_min[idx] * imax for idx, imax in enumerate(int_res_max) ]
    print(ea.member_selector)

    df_ens = main_eval_offline_ensemble(ea)
    ens_perf_fn='agent_perf/ensemble_agent_perf.' + env_selector + '.csv'
    storage.write_df_to_location( df_ens, ens_perf_fn)

    df_all = df_mem.copy()
    df_all = df_all.append(df_ens)

    all_perf_fn = 'agent_perf/all_agent_perf.' + env_selector + '.csv'
    storage.write_df_to_location(df_all, all_perf_fn)
    print(df_all)



def call_full_process():
    main_generate_replay()
    main_train_offline()
    ea=get_ensemble_agent()
    main_eval_offline_members(ea)
    main_eval_offline_ensemble(ea)


def call_top(cmd=None, config="config/bandit.yaml"):
    global conf

    # config ---
    conf_file_default = 'config/default.yaml'
    with open(conf_file_default) as f:
        conf_default = yaml.load(f, Loader=yaml.FullLoader)
        #print(conf_default)

    conf_file = config #'config/cartpole.yaml'
    if conf_file == conf_file_default:
        conf = conf_default
    else:
        with open(conf_file) as f:
            conf_over = yaml.load(f, Loader=yaml.FullLoader)
            #print(conf_over)

        conf = conf_default
        conf.update(conf_over)

    global env_selector
    env_selector = conf['label']

    settings()

    if cmd=="rollout":
        main_generate_replay()
    elif cmd == "train":
        main_train_offline()
    elif cmd=="full":
        call_full_process()
    elif cmd=="eval_members":
        call_eval_members()
    elif cmd=="eval_ensemble":
        call_eval_ensemble()
    else:
        print("Unknown cmd")

import fire

if __name__ == '__main__':
    fire.Fire(call_top)

# replay explorer?
