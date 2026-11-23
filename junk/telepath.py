# slightly ugly way to turn off DeprecationWarnings
silence_deprecation_warnings = True
if silence_deprecation_warnings:
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

import gym
import random
import numpy as np
import pandas as pd
import torch as th
import sys
import boto3
from pprint import pprint as pp
import yaml
import fire
from telepath_worker import telepath_utils
from telepath_worker.telepath_utils import dbg
import os
import subprocess
import telepath_worker.spec
import time

MLFLOW=0
if MLFLOW:
    import mlflow.pytorch
    import mlflow

PYTORCH_DEVICE='cuda:0'
#PYTORCH_DEVICE='cpu'

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN, BatchDQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

#import gym_stochastic.envs.simplebandits

from ensemble_agent import EnsembleAgent
from evaluator import evaluate_eps
from ext_replay import PdPlainBuffer, VxBuffer
import storage_layer as storage
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    EveryNTimesteps,
    StopTrainingOnRewardThreshold,
)


ENABLE_WAND=0
if ENABLE_WAND:
    import wandb

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


execution_id = None

class TelepathEngine():

    conf=None
    sett = {}           # dict values are not objects, just simple values
    sett_algo = {}      # algo config: dict values are not objects, just simple values
    sett_dynamic = {}   # dont try to put these in tables
    ea = None

    #BUFFER_TYPE = PdFrameBuffer
    RW_BUFFER_TYPE = PdPlainBuffer
    READ_BUFFER_TYPE = PdPlainBuffer #VxBuffer

    # project-wide
    tel_project_dict = {}

    # project-specific
    tel_proj_buffer_dict = {}
    tel_proj_agent_dict = {}
    tel_proj_train_dict = {}
    tel_proj_eval_dict = {}
    tel_proj_train_dict = {}
    tel_proj_predict_dict = {}


    def __init__(self, config=None):
        self.telepath_init( config )
        self.tel_project_dict[ self.conf['label'] ] = {}
        execution_id = telepath_utils.get_rand_execid()

    def telepath_init(self, config="config/bandit.yaml"):

        # config ---

        # merge default.yaml, with whatever config
        conf_file_default = 'config/default.yaml'
        with open(conf_file_default) as f:
            conf_default = yaml.load(f, Loader=yaml.FullLoader)
            #print(conf_default)

        conf_file = config #'config/cartpole.yaml'
        if conf_file == conf_file_default:
            self.conf = conf_default
        else:
            with open(conf_file) as f:
                conf_over = yaml.load(f, Loader=yaml.FullLoader)
                #print(conf_over)

            self.conf = conf_default
            self.conf.update(conf_over)

        global env_selector
        env_selector = self.conf['label']

        self.process_settings()
        self.sett['config_file']=config
        self.sett['algo_config_file']=config


    def process_settings(self):
        # awkward place for these?
        # or perfect place for constructed settings.
        storage.ensure_folder_exists(self.conf['folder'] + "agents/" )
        self.sett['online_agent_path'] = self.conf['folder'] + "agents/online_agent.bin"

        # replay buffers
        #self.sett['online_agent_replay_path'] = self.conf['folder'] + "replay/online_agent_replay"
        storage.ensure_folder_exists(self.conf['folder'] + "replay/" )
        self.sett['replay_loc'] = self.conf['folder'] + "replay/full_replay.csv"  # extended_replay4.csv" #
        #self.sett['replay_loc'] = self.conf['folder'] + "replay/extended_replay4.csv"

        # agent performance dfs
        storage.ensure_folder_exists(self.conf['folder'] + "agent_perf/" )
        self.sett['mem_perf_fn'] = self.conf['folder'] + 'agent_perf/df_mem.csv'
        self.sett['ens_perf_fn']=self.conf['folder'] + 'agent_perf/ensemble_agent_perf.csv'
        self.sett['all_perf_fn'] = self.conf['folder'] + 'agent_perf/all_agent_perf.' + env_selector + '.csv'

        # basically a global
        self.sett_dynamic['vec_env'] = DummyVecEnv([lambda: gym.make( self.conf['env_label'] )])

        self.sett['rw_replay'] = self.RW_BUFFER_TYPE(self.sett['replay_loc'])
        self.sett['rw_replay'].buffer_size = 50000  # TODO: weird that this is set here.

        self.sett['read_replay'] = self.READ_BUFFER_TYPE(self.sett['replay_loc'])
        self.sett['read_replay'].buffer_size = 50000  # TODO: weird that this is set here.
        return self.sett


    # -- agent mgmt
    # agents need to be parameterized much better.
    # agent manager?

    def log_execution(self, command, msg = ""):
        dbg('log_execution','execution_id: '+str(execution_id)+ ' : '+ command + ' / ' + msg)
        # should write log file

    def get_batch_agent_fn(self,env_selector, member_index):
        dbg("get_batch_agent_fn", data=[env_selector, member_index])
        batch_agent_fn = self.conf['folder'] + 'agents/batch_agent.' + env_selector + '.' + str(member_index) + '.telepath'
        return batch_agent_fn


    def get_ensemble_agent(self):
        dbg("get_ensemble_agent")
        ensemble = EnsembleAgent(self.sett_dynamic['vec_env'])
        for es_idx in range(0,self.conf['ensemble_size']):
            batch_agent_fn = self.get_batch_agent_fn( env_selector, es_idx )
            #batch_agent = BatchDQN.load(batch_agent_fn)
            batch_agent = storage.load_agent_location( BatchDQN, batch_agent_fn)
            ensemble.members.append( batch_agent )
        return ensemble







    #def main_train_online(self):


    def main_generate_replay(self, vec_env, replay_out ):

        if ENABLE_WAND:
            wandb.init(sync_tensorboard=True, project="tel3-" + self.conf['label'], dir=self.conf['folder']+'/')

        # --- try loading
        dbg("generate_replay",comment="create agent and buffer")

        eval_callback = EvalCallback(
            eval_env=vec_env,
            # callback_on_new_best=callback_on_best,
            best_model_save_path='agent_best_save',
            log_path=self.conf['tblog_folder'],
            eval_freq=1000,
            n_eval_episodes = 2,
        )

        #online_agent = DQN('MlpPolicy', env, learning_starts=1000, verbose=1, learning_rate=0.001)
        online_agent = DQN('MlpPolicy', vec_env, gamma=self.conf['env_gamma'],
                target_update_interval=self.conf['agent_target_update_interval'],
                buffer_size = 1000000,
                verbose=0,
                exploration_fraction = 0.5,
                learning_starts=self.conf['agent_learning_starts'],
                learning_rate=self.conf['agent_learning_rate'],
                batch_size=self.conf['agent_batch_size'], tensorboard_log=self.conf['tblog_folder'],
                device=PYTORCH_DEVICE,
                )
        online_agent.agentid = telepath_utils.get_rand_agentid()

        res=evaluate_eps( vec_env, 'online_agent', online_agent, eval_eps=self.conf['eval_eps']  ); print("online_agent before learn",res)
        print("online_agent.replay_buffer.pos",online_agent.replay_buffer.pos)

        for steps in [ self.conf['total_samples'],]: #20000]:
        #for steps in [20, 200]:
            online_agent.learn( log_interval=4,
                                total_timesteps=steps,
                                eval_env=vec_env,
                                #callback=eval_callback,
                                tb_log_name=self.conf['label'] + '_dqn' )    # log_interval is in episodes
            print("online_agent.replay_buffer.pos",online_agent.replay_buffer.pos)
            res=evaluate_eps( vec_env, 'online_agent', online_agent, eval_eps=self.conf['eval_eps']  ); print("online_agent after learn",res)

        for i in range(0,self.conf['eval_reps']):
            res = evaluate_eps(vec_env, 'online_agent', online_agent, eval_eps=self.conf['eval_eps'] ); print("online_agent after learn", res)

        #sys.exit()
        #online_agent.save(online_agent_path)
        storage.save_agent_location( online_agent,self.sett['online_agent_path'] )
        #online_agent.save_replay_buffer(self.sett['online_agent_replay_path'])   # native sbl3 buffer -- needed?

        dbg("generate_replay",comment="transfering FULL replay... now fast with lazy_add")
        onr = online_agent.replay_buffer

        for i in range(0,onr.pos):
            batch_inds = [i]
            self.sett['rw_replay'].lazy_add(
                obs=onr.observations[batch_inds, 0, :][0],
                next_obs=onr.next_observations[batch_inds, 0, :][0],
                action=onr.actions[batch_inds, 0, :][0],
                reward=onr.rewards[batch_inds][0],
                done=onr.dones[batch_inds][0],
            )
            #if i % 500 ==0: print(i)

        self.sett['rw_replay'].reconstruct()

        dbg("generate_replay",comment="save replay")
        # full_replay.df.to_feather('full_replay.fth',)
        #storage.write_pickle_to_location( self.sett['full_replay'].df, self.sett['replay_loc'])

        #if 'batch_delete_actions' in self.conf:
        if True:  # TODO: Turn off filter!
            #filtered_actions = 'batch_delete_actions'
            #filtered_actions=[1,3]
            #print(filtered_actions)
            #exit(1)
            self.sett['rw_replay'].df = self.sett['rw_replay'].df[ ~( (self.sett['rw_replay'].df.action.isin( [1] )) & ( self.sett['rw_replay'].df.obs0.isin( [2.0] )  ))  ]

        storage.write_df_to_location( self.sett['rw_replay'].df, replay_out )
        ## self.sett['rw_replay'].write()

    '''
    Current replay overview:
    1) regular replay is copied into PdPlainBuffer (one row at a time)
    2) batch agents get their replay overwritten with this one
    
    What I want:
    1) VBuffer is populated in batch by regular replay
    2) batch agents get vbuffer
    '''




    def main_train_offline( self, replay_loc ):

        dbg("main_train_offline", comment="load replay", data=replay_loc)
        #self.sett['read_replay'].df = storage.read_df_from_location(self.sett['replay_loc'])
        #self.sett['read_replay'].read_from_location(self.sett['replay_loc'])

        dbg("main_train_offline", comment="create ensemble, train all members",)

        # create replay buffer
        common_replay = self.READ_BUFFER_TYPE(replay_loc)  # why two?



        if ENABLE_WAND:
            wandb.init(sync_tensorboard=True, project="tel3-" + self.conf['label'], dir=self.conf['folder']+'/')

        eval_callback = EvalCallback(
            self.sett_dynamic['vec_env'],
            # callback_on_new_best=callback_on_best,
            #best_model_save_path='agent_best_save',
            log_path=self.conf['tblog_folder'],
            eval_freq=1000,
            n_eval_episodes=1,
        )


        ea = EnsembleAgent(self.sett_dynamic['vec_env'])

        for es in range(0,self.conf['ensemble_size']):
            batch_agent = BatchDQN('MlpPolicy', self.sett_dynamic['vec_env'], verbose=2, learning_rate=self.conf['agent_learning_rate'], gamma=self.conf['env_gamma'],
                        batch_size=self.conf['agent_batch_size'], buffer_size=10, target_update_interval=self.conf['agent_target_update_interval'],
                        learning_starts=self.conf['agent_learning_starts'], device=PYTORCH_DEVICE, tensorboard_log=self.conf['tblog_folder'], )
            batch_agent.agentid = telepath_utils.get_rand_agentid()

            del batch_agent.replay_buffer

            # install replay buffer
            batch_agent.replay_buffer = common_replay
            res=evaluate_eps( self.sett_dynamic['vec_env'], 'batch_agent', batch_agent, eval_eps=self.conf['eval_eps'] ); print("batch_agent before learn", res)

            #print(self.sett['read_replay'].df)

            # link to replay buffer (is this needed?)
            #batch_agent.replay_buffer.df = self.sett['read_replay'].df #.sample(total_samples)
            batch_agent.replay_buffer = self.sett['read_replay'] #.sample(total_samples)

            #batch_agent.replay_buffer.copy_from( online_agent.replay_buffer )
            #print("batch_agent.replay_buffer.pos",batch_agent.replay_buffer.pos)
            #collect_rollouts( batch_agent, env, batch_agent.replay_buffer, steps=1000 )

            trainsteps=0
            trainsteps_add = self.conf['train_steps_per_agent']

            dbg("main_train_offline", comment="learn_batch", )
            batch_agent.learn_batch(total_timesteps=trainsteps_add,
                                    tb_log_name= self.conf['label'] + '_batch_dqn',
                                    #callback=eval_callback
                                    )

            dbg("main_train_offline", comment="batch_agent.replay_buffer.pos", data= batch_agent.replay_buffer.pos )
            res = evaluate_eps(self.sett_dynamic['vec_env'], 'batch_agent',batch_agent, eval_eps=self.conf['eval_eps'] );
            dbg("main_train_offline", comment="batch_agent after learn", data= res )

            trainsteps += trainsteps_add
            dbg("main_train_offline", comment="trainsteps", data= trainsteps )

            # add the trained member to the ensemble
            ea.members.append( batch_agent )

            batch_agent_fn = self.get_batch_agent_fn( env_selector, es )
            dbg("main_train_offline", comment="save_agent_location", data= batch_agent_fn )
            storage.save_agent_location( batch_agent, batch_agent_fn )

        # tear down ensemble;
        # its very odd what happened when I didnt do this:
        # get_ensemble_agent then already had (seemlingly valid agent) members on construction.
        while ea.members:
            ag = ea.members[-1]
            #print('del ',ag.agentid)
            del ea.members[-1]
        del ea



    def main_eval_offline_members(self,vec_env, ea):

        dbg("main_eval_offline_members", )
        df=pd.DataFrame()
        for idx,ag in enumerate( ea.members):
            print(" member ",str(idx))
            for i in range(0,self.conf['eval_reps']):
                res = evaluate_eps(vec_env, 'composite_member_'+str(i), ag, eval_eps=self.conf['eval_eps'] )
                res.update({ 'member': str(idx)  })
                df = df.append( res,ignore_index=True )
                #print("member agent", res)
        return df



    def main_eval_offline_ensemble(self,vec_env,ea):
        dbg("main_eval_offline_ensemble", )

        df=pd.DataFrame()

        #print(" + mode: thompson",)
        for i in range(0, self.conf['eval_reps']):
            res = evaluate_eps(vec_env, 'composite_agent',ea, deterministic=False, eval_eps=self.conf['eval_eps'] );
            res.update({'mode': 'thompson'})
            df = df.append(res, ignore_index=True)
            #print("thompson agent", res)

        low_modes = [('min','min'),('min','mean'),('min','std'),('max','std'),]
        all_modes = [('max','median'),('max','mean',),('max','min'),('max','max'),]
        modes = [('max','mean',),('max','median'),]

        for mode_selector,mode_ens in all_modes:
            #print(" + mode: ", mode_selector,mode_ens)
            for i in range(0,self.conf['eval_reps']):

                ea.mode_selector = mode_selector
                ea.mode_ens = mode_ens

                res = evaluate_eps(vec_env, 'composite_agent', ea, eval_eps=self.conf['eval_eps']);
                res.update({'mode_ens': mode_ens, 'mode_selector': mode_selector,'mode': mode_selector+'_'+mode_ens  })
                df = df.append( res,ignore_index=True )
                #print("ensemble agent", res)

        #print(df)
        agent = ea.members[0]
        obs = vec_env.reset()
        eps_count = 0
        needs_last_reward = False
        deterministic = True

        for i in range(0,10):
            action, q_values, state = agent.predict_raw(obs,)
            next_obs, reward, done, info = vec_env.step(action)
            print("obs q_values action reward done next_obs", obs, q_values, action, reward, done, next_obs)
            if done:
                obs = vec_env.reset()
            else:
                obs=next_obs


    '''
    Concept: partial buffer
    
    training buffer: action, rewards, obs, next_obs
    policy buffer: (sampleid), action, [obs]
    query buffer: (sampleid), obs
    
    '''


    def main_predict(self, agent_map, in_loc=None, out_loc=None):
        # load agent
        #in_loc = 'simplebandit/predict/predict_in.csv'
        #out_loc='predictions.csv'

        dbg("main_predict",data=[ in_loc, out_loc ] )

        buf=PdPlainBuffer(in_loc)

        df_in = pd.read_csv(in_loc)
        # TODO: make this work for:
        # - huge files
        # - s3 files
        # - "replay" files with specs

        #df_in = storage.read_df_from_location(in_loc) # pd.read_csv(in_loc)

        '''
        What if this csv was:
        - on s3
        - a different name
        - a non-csv type buffer
        
        Rather be:

        main_predict ( agent, read_buffer_in, write_buffer_out ):  # subclass of InputBuffer

        read_buffer_in.connect()
        write_buffer_out.connect()  # initialize a new file

        while True:
            sample = read_buffer_in.next(batch_size)
            action, _states = agent.predict(obs, deterministic=deterministic)
            write_buffer_out.add( sample, action )

        write_buffer_out.close()
        read_buffer_in.close()
        '''

        print(df_in.head(2))

        df_out = pd.DataFrame()

        #def evaluate_eps(env, agent_label, agent, eval_eps=10, deterministic=True, ):
        if MLFLOW:
            mlflow.set_experiment('PolicyEval')
            mlflow.set_tracking_uri(self.conf['mlflow_folder'])
            #with mlflow.start_run(run_name=agent_label, ) as run:

        start_index = 0
        batch_size = 1  # only batch size 1 works for now
        start = start_index

        # how to determine obs columns?
        # in_loc_spec_path = None,
        # rspec = spec.ReplaySpec(in_loc_spec_path)
        # rspec.load()

        # in_loc_spec_path
        obs_cols = [ 'obs0', 'obs1', 'obs2', 'obs3', ]
        deterministic = True

        iters = 0

        while True:
            s_time = time.time()

            # determine sample window
            end = start + batch_size
            # went past the end?
            if end > df_in.shape[0] - 1:
                print("done")
                break
            # print(" sample " ,start,end)
            s1_time = time.time()
            sample = df_in.iloc[start:end]
            # todo: handle actual batches
            e1_time = time.time()
            d1_time = e1_time - s1_time

            start += batch_size

            # load sample into pytorch
            sid = sample.iloc[0].sid

            #sid = genutil.get_rand_sampleid()
            obslist= [  th.tensor(sample[oc].tolist(),dtype=th.float)  for oc in obs_cols ]
            obs = th.stack( obslist, dim=1,)

            row_dict = {'sid':sid, }

            # agent prediction
            for agent_key in agent_map.keys():
                action, _states = agent_map[agent_key].predict(obs, deterministic=deterministic)
                row_dict[agent_key] = int(action[0])

            # prediction storage

            # if append_obs:
            #
            # row_dict[obs_col]= obslist[idx]

            df_out=df_out.append( row_dict, ignore_index=True)

            # print(action)
            iters+= 1
            if iters %500==0:
                print(iters)

        for agent_key in agent_map.keys():
            #print( agent_key )
            print( df_out.groupby(agent_key).count() )

        df_out.to_csv(out_loc,index=False)





    def call_eval_members(self, vec_env):
        ea = self.get_ensemble_agent()
        ea_size = len(ea.members)
        df_mem = self.main_eval_offline_members(vec_env,ea)
        storage.write_df_to_location(df_mem, self.sett['mem_perf_fn'])



    def call_eval_ensemble(self, vec_env):
        ea = self.get_ensemble_agent()
        try:
            df_mem = storage.read_df_from_location(self.sett['mem_perf_fn'])
        except:
            df_mem = pd.DataFrame()
        df_ens = self.main_eval_offline_ensemble(vec_env,ea)
        storage.write_df_to_location( df_ens, self.sett['ens_perf_fn'])



    def _partial_ensemble(self):
        '''
        Look at ways to select best sub-ensemble
        '''
        ea = self.get_ensemble_agent()

        try:
            df_mem = storage.read_df_from_location(self.sett['mem_perf_fn'])
            print("Perf data loaded:",self.sett['mem_perf_fn'])
        except:
            print("Error: Perf data not found:",self.sett['mem_perf_fn'])
            df_mem = pd.DataFrame()
            return
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

        df_ens = self.main_eval_offline_ensemble(ea)
        storage.write_df_to_location( df_ens, self.sett['ens_perf_fn'])

        df_all = df_mem.copy()
        df_all = df_all.append(df_ens)

        storage.write_df_to_location(df_all, self.sett['all_perf_fn'])
        print(df_all)






    def call_full_process(self, vec_env, replay_loc):

        # combine env-training and rollout collection
        self.main_generate_replay(vec_env, replay_loc)

        self.main_train_offline( replay_loc )

        ea=self.get_ensemble_agent()

        df_mem=self.main_eval_offline_members(vec_env,ea)

        df_ens=self.main_eval_offline_ensemble(vec_env,ea)

        df_tot = df_mem
        df_tot=df_tot.append(df_ens,ignore_index=True)
        print(df_tot)


def get_env_from_spec( env_id ):
    vec_env = DummyVecEnv([lambda: gym.make(env_id)])
    # vec_env = gym.make( env_id )
    return vec_env


def proc_exec( bash_list, capture_output=True ):
    if capture_output:
        proc = subprocess.run( bash_list, capture_output=capture_output )
    else:
        proc = subprocess.run( bash_list, capture_output=capture_output, stderr=sys.stderr, stdout=sys.stdout)
    return(proc.stdout.decode('ascii'))


class TelepathCmd():

    te = None

    def __init__(self, config): # init with config file??
        self.te = TelepathEngine( config )
        #self.status()

    def replay_stats(self, replay_loc):
        read_replay = PdPlainBuffer(replay_loc)
        print( read_replay )
        read_replay.buffer_stats()


    def train_online(self, env_id, ): # agent_dest, train_config): # agent_dest, env_src, train_online params
        '''
        in: env spec
        out: replay buffer ; trained online agent
        '''
        #vec_env = get_env_from_spec( env_id )
        self.te.main_generate_replay( self.te.sett_dynamic['vec_env'], self.te.sett['replay_loc'] )


    def rollout(self, env_id): # agent_src, env_src, rollout params, replay_location
        '''
        in: env spec
        out: replay buffer ; trained online agent
        '''
        self.te.main_generate_replay(self.te.sett_dynamic['vec_env'], self.te.sett['replay_loc'])


    def train_offline(self, train_file ): # replay_src, env_id, rollout params, agent_dest
        '''
        in: replay buffer
        out: trained agents
        '''
        self.te.main_train_offline( train_file )


    def predict(self, agent_path, in_features_path, out_pred_path ):     #in_features_path, out_pred_path):
        #in_features_path = 's3://example-telepath-bucket/predict/full_replay.csv'
        #out_pred_path = 's3://example-telepath-bucket/predict/out.csv'
        '''
        in: obs input, agent
        out: prediction output
        '''

        #agent_spec_file
        # ensemble_agent = self.te.get_ensemble_agent()
        #batch_agent_fn = self.get_batch_agent_fn(env_selector, es_idx)
        # batch_agent = BatchDQN.load(batch_agent_fn)
        # batch_agent_fn = '/path/to/telepath-runs/cartpole/agents/batch_agent.cartpole.0.telepath'
        batch_agent = storage.load_agent_location(BatchDQN, agent_path)

        # TODO: should specific the agent itself

        agent_map = {'batch_agent':batch_agent,} #'ensemble_agent':ensemble_agent}
        #agent_map = {'batch_agent':batch_agent, 'ensemble_agent':ensemble_agent}
        self.te.main_predict( agent_map, in_features_path, out_pred_path )


    def predict_test(self,  ):     #in_features_path, out_pred_path):
        in_features_path = 's3://example-telepath-bucket/predict/full_replay.csv'
        out_pred_path = 's3://example-telepath-bucket/predict/out.csv'
        agent = self.te.get_ensemble_agent()
        self.te.main_predict( agent, in_features_path, out_pred_path )


    def eval_members(self):
        '''
        in: agent, env spec
        out: eval
        '''
        self.te.call_eval_members(self.te.sett_dynamic['vec_env'])


    def eval_ensemble(self):
        '''
        in: agent, env spec
        out: eval
        '''
        self.te.call_eval_ensemble( self.te.sett_dynamic['vec_env'] )


    def fulltest(self):
        self.te.call_full_process( self.te.sett_dynamic['vec_env'], self.te.sett['replay_loc'] )


    def info(self):
        print("Telepath v0.7.8")
        print("Copyright (c) 2020 Pathway Intelligence Inc.")


    def tb(self):
        tb_log = self.te.conf['tblog_folder']
        print("Running tensorboard, visit port 6006")
        print( proc_exec(['tensorboard','--logdir='+tb_log ], capture_output=False) )
        #subprocess.run(['ls'], stderr=sys.stderr, stdout=sys.stdout)


    def status(self):
        self.info()
        print('torch.__version__',th.__version__)
        print('torch.version.cuda',th.version.cuda)
        print( proc_exec(['uname','-a']) )
        print( proc_exec(['cat','/etc/issue']) )
        print( proc_exec(['free','-h']) )
        print( proc_exec(['lscpu']) )
        print( proc_exec(['nvidia-smi']) )
        pp(self.te.conf)
        pp(self.te.sett)
        #print( proc_exec(['ps','axufwww']) )
        #print( proc_exec(['df','-h']) )
        #print( proc_exec['top -n1'] )


if __name__ == '__main__':
    fire.Fire(TelepathCmd)


'''
Use cases
---------

train_offline ---

python telepath.py 
    train_offline 
        --train_file=s3://example-telepath-bucket/predict/full_replay.csv 
        --config=config/default.yaml
            # dqn train params
            # env params
            # other params...

predict ---

python telepath.py predict --in_features_path=s3://example-telepath-bucket/predict/full_replay.csv --out_pred_path=s3://example-telepath-bucket/predict/out.csv  --config=config/default.yaml



evaluate ---



'''

