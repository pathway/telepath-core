#!/usr/bin/env python3

#slightly ugly way to turn off DeprecationWarnings
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
import vaex
import plotille
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import sklearn.tree
from sklearn.metrics import classification_report
import functools
import time
import datetime
import json
from telepath_common import *

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"[Elapsed time] {elapsed_time:0.3f} sec")
        return value
    return wrapper_timer


os.environ['NUMEXPR_MAX_THREADS']=str(8)

MLFLOW=0
if MLFLOW:
    import mlflow.pytorch
    import mlflow

version_file = os.path.join(os.path.dirname(__file__), ".", "version.txt")
with open(version_file, "r") as file_handler:
    telepath_version = file_handler.read().strip()


from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import pdb
#import gym_stochastic.envs.simplebandits

from telepath_worker.evaluator import evaluate_eps
from telepath_worker.spec import ReplaySpec, AlgoSpec, SettingsSpec, ReplayPrepSpec
from telepath_worker.ext_replay import PdPlainBuffer, VxBuffer, get_specgym_from_replay_spec
import telepath_worker.storage_layer as storage
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


def get_vec_env_from_label( gym_label):
    ve = DummyVecEnv([lambda: gym.make( gym_label )])
    return ve

def get_vectorize_env( gym_obj ):
    ve = DummyVecEnv([ lambda: gym_obj ])
    return ve



from telepath_agent_sbl3 import *


class TelepathEngine():

    setspec=None
    execution_id=None

    def telepath_init(self, config):
        self.setspec = SettingsSpec()
        self.setspec.load(config)
        self.execution_id=telepath_utils.get_rand_execid()

    def log_execution(self, command, msg = ""):
        dbg('log_execution','execution_id: '+str(self.execution_id)+ ' : '+ command + ' / ' + msg)
        # should write log file

    def get_replay_read(self, replay_loc_path, replay_spec_path):
        # doesnt need to read twice?
        replay_spec = ReplaySpec()
        #replay_spec_path = 'specs2/spec_replay_cartpole_tiny.yaml'
        replay_spec.load(replay_spec_path)
        #print(replay_spec.spec)

        #replay_loc = '/path/to/telepath-runs/cartpole_tiny/replay/full_replay.csv'
        replay = PdPlainBuffer(replay_loc_path)
        replay.set_spec(replay_spec_path)
        #print(replay.specs.spec)
        return replay


    def main_generate_replay(self, vec_env_sim, write_replay_path, write_replay_spec_path, algo_spec, use_tid=False, online_agent_dest_path=None ):

        if ENABLE_WAND:
            wandb.init(sync_tensorboard=True, project="tel3-" + self.conf['label'], dir=self.conf['folder']+'/')

        # --- try loading
        dbg("generate_replay",comment="create agent and buffer")

        eval_callback = EvalCallback(
            eval_env=vec_env_sim,
            # callback_on_new_best=callback_on_best,
            #best_model_save_path='agent_best_save',
            log_path=self.setspec.spec['content']['tblog_folder'],
            eval_freq=1000,
            n_eval_episodes = 2,
        )
        eval_callback = None

        #print(algo_spec)
        #print(algo_spec.spec['content'])
        algo_ = algo_spec.spec['content']
        set_=self.setspec.spec['content']

        telepath_agent_online = TelepathAgent_DiscretaAction_Offpolicy_Sbl3()
        telepath_agent_online._init_agent( algo_, set_, vec_env_sim, batch_online="ONLINE", )

        res=self.evaluate_eps( vec_env_sim, 'telepath_agent_online', telepath_agent_online, status="before learn");

        print("telepath_agent_online.get_buffer_pos",telepath_agent_online.get_buffer_pos() )

        #execution_id = None

        params = {
            'log_interval': 10,
            'total_timesteps': algo_['total_samples'],
            'eval_env': vec_env_sim,
            # callback=eval_callback,
            'tb_log_name': 's_dqn_' + telepath_agent_online.getid()
        }
        telepath_agent_online.learn_online( params  )


        print("telepath_agent_online.get_buffer_pos",telepath_agent_online.get_buffer_pos() )
        res=self.evaluate_eps( vec_env_sim, 'telepath_agent_online', telepath_agent_online, status="after learn" );

        dbg("generate_replay",comment="transfering replay")
        onr = telepath_agent_online.get_buffer()

        replay_write = self.get_replay_read(replay_loc_path=write_replay_path, replay_spec_path=write_replay_spec_path)

        for i in range(0,onr.pos):
            batch_inds = [i]
            #pp( onr.observations[batch_inds, 0, :][0] )
            sids = None
            tids = []
            steps = None
            #if use_tid:
            #    tids = []

            replay_write.lazy_add(
                obs=onr.observations[batch_inds, 0, :][0],
                next_obs=onr.next_observations[batch_inds, 0, :][0],
                action=onr.actions[batch_inds, 0, :][0],
                reward=onr.rewards[batch_inds][0],
                done=onr.dones[batch_inds][0],
                sids=sids,
                tids=tids,
                steps=steps
            )
            #if i % 500 ==0: print(i)

        replay_write.reconstruct()

        print(list ( replay_write.df ) )

        dbg("generate_replay",comment="save replay",data=write_replay_path)
        storage.write_df_to_location( replay_write.df, write_replay_path )

        if online_agent_dest_path:
            print("Saving online agent to", online_agent_dest_path )
            telepath_agent_online.save_agent( online_agent_dest_path )

    def get_replay_summary(self, df ):
        #print(df.describe())
        stat_cols= df.describe().T.sort_index()
        #print(stat_cols)
        stat_cols['isnull'] = df.isnull().sum(axis=0)
        stat_cols['notnull'] = df.notnull().sum(axis=0)
        #print(stat_cols)

        stat_cols.sort_index( inplace=True)
        #stat_cols['sum'] = df.sum(axis=0)
        return stat_cols

    def main_prepare_replay(self, read_replay_path, write_replay_path, replay_prep_spec_path, prep_mode, debug_prep=True ):
        #  write modified csv.
        dbg("main_prepare_replay","args")
        valid_prep_mode = ['train','predict']
        if prep_mode not in valid_prep_mode:
            print("prep_mode: {} not in {}".format( prep_mode, valid_prep_mode ) )
            exit(1)
        pp({'prep_mode':prep_mode,
            'read_replay_path':read_replay_path,
           'write_replay_path':write_replay_path,
           'replay_prep_spec_path': replay_prep_spec_path})

        def expand_fieldnames( fldsets, fldlist, _dbg=0 ):
            if _dbg: print("[expand_fieldnames] got",fldsets, fldlist)
            final_list=[]
            for f in fldlist:
                if f in fldsets.keys():
                    if _dbg: print("[expand_fieldnames] appending", fldsets[f],type(fldsets[f]))
                    final_list += fldsets[f]
                else:
                    final_list.append(f)
            if _dbg: print("[expand_fieldnames] return",fldsets, fldlist, final_list)
            return final_list

        # TODO: make this scalable to very large files.  Make test case.

        # TODO: creation of next_obs_ columns via shift operations.


        # --- try loading
        dbg("main_prepare_replay",comment="create replay from raw train set")

        replay_prep_spec = ReplayPrepSpec()
        replay_prep_spec.load( replay_prep_spec_path )

        # create replay buffer
        if 0: # cheat
            read_replay_df = storage.read_df_from_location(read_replay_path)
            storage.write_df_to_location( read_replay_df,write_replay_path)
            return

        # this was causing issues when replay changed but hdf5 cached file did not
        #hdf5_fn = read_replay_path + '_temp.hdf5'
        #dbg("main_prepare_replay","Temp hdf5 fn",hdf5_fn)

        dbg("prep","read_replay_path",read_replay_path)

        df = pd.read_csv(read_replay_path) #, chunk_size=50_000)  # convert=hdf5_fn,

        #d = self.get_replay_summary( df )
        #print(d )

        #print(read_replay_path)
        #import pdb; pdb.set_trace()

        cont = replay_prep_spec.spec['content']

        fldsets={}
        if 'fldsets' in cont:
            fldsets_dict = cont['fldsets']
            for k in fldsets_dict.keys():
                fldsets[k] = fldsets_dict[k].split(' ')
                print(fldsets)
            #fldsets = { for f in fldsets_str.split(' ') if f!='' and f is not None]
            print('fldsets',fldsets)

        #print(df)
        obs_stats = {}
        pd.set_option('display.max_rows', 40)

        prep_cont = cont['prep_scripts']
        print(prep_cont)
        print("**** prep_scripts :")
        for ln,xf in enumerate(prep_cont):
            print(str(ln)+'  ')
            print(xf)

        for top_line_num,topmost_line in enumerate(prep_cont):

            print("top line: ", top_line_num,topmost_line)

            pm = prep_cont[top_line_num]['prep_mode']
            if pm != 'all' and pm!=prep_mode:
                print(' prepmode no match, skipping',pm )
                continue

            for line_num,cont_line in enumerate(prep_cont[top_line_num]['script']):

                #print("inner line: ", line_num,cont_line)

                if debug_prep:
                    print('   before columns: ',list(df))
                    print( self.get_replay_summary(df) )
                    print('---')


                cmd_toks = cont_line.split(' ')
                cont_key = cmd_toks.pop(0)

                print('** prep_script line {}:'.format(line_num),cont_line,)


                # --- ELEMENTAL

                if cont_key == 'log':
                    flds = expand_fieldnames( fldsets, cmd_toks )
                    for src_fld in flds:
                        new_fld = src_fld+'_log'
                        df[ new_fld ] = np.log10(df[ src_fld ])
                        df[new_fld] = df[ new_fld ].fillna(0.0).astype(np.float32)

                elif cont_key == 'logp1':
                    flds = expand_fieldnames( fldsets, cmd_toks )
                    for src_fld in flds:
                        new_fld = src_fld+'_log'
                        df[ new_fld ] = np.log10(df[ src_fld ]+1.0)
                        df[new_fld] = df[ new_fld ].fillna(0.0).astype(np.float32)
                    #import pdb; pdb.set_trace()

                elif cont_key == 'f16':
                    flds = expand_fieldnames( fldsets, cmd_toks )
                    for src_fld in flds:
                        df[src_fld] = df[ src_fld ].astype(np.float16)

                elif cont_key == 'f32':
                    flds = expand_fieldnames( fldsets, cmd_toks )
                    for src_fld in flds:
                        df[src_fld] = df[ src_fld ].astype(np.float32)

                elif cont_key == 'i32':
                    flds = expand_fieldnames( fldsets, cmd_toks )
                    for src_fld in flds:
                        df[src_fld] = df[ src_fld ].astype(np.int32)

                elif cont_key == 'fillna0':
                    flds = expand_fieldnames( fldsets, cmd_toks )
                    for src_fld in flds:
                        df[src_fld] = df[ src_fld ].fillna(0.0)

                elif cont_key == 'bool':
                    flds = expand_fieldnames( fldsets, cmd_toks )
                    for src_fld in flds:
                        df[src_fld] = df[src_fld].fillna(0).replace({True: 1, False: 0})

                elif cont_key == 'noop':
                    pass

                elif cont_key == 'const':
                    fld = cmd_toks.pop(0)
                    val = float(cmd_toks.pop(0))
                    df[fld] = val

                # --- FIELD SELECTION

                elif cont_key == 'rm':
                    flds = expand_fieldnames( fldsets, cmd_toks )
                    df = df.drop( columns = flds )

                elif cont_key == 'sel':
                    flds = expand_fieldnames( fldsets, cmd_toks )
                    df = df[ flds ]


                # --- TWO FIELDS

                elif cont_key == 'mult_f':
                    mult_fld = cmd_toks.pop(0)
                    flds = expand_fieldnames( fldsets, cmd_toks )
                    for src_fld in flds:
                        df[ src_fld ] = df[ src_fld ]  * df[ mult_fld ]


                # --- * STATEFUL : affected by overall stats

                elif cont_key == 'std':
                    flds = expand_fieldnames( fldsets, cmd_toks )
                    for idx, oc in enumerate(flds):
                        dic=d[oc].to_dict()
                        obs_stats[oc] = {'mean': dic['mean'], 'std': dic['std']}
                        df[oc] = (( df[oc] - obs_stats[oc]['mean'] ) /  obs_stats[oc]['std']).astype(np.float32)

                elif cont_key == 'ren':
                    df = df.rename( columns={ cmd_toks[0]: cmd_toks[1] } )


                # --- SORTING

                elif cont_key == 'sort':
                    sort_list = [ s for s in cmd_toks if s!='' and s!=None  ]
                    df = df.sort_values(by=sort_list)

                elif cont_key == 'sort_rand':
                    df = df.sample(frac=1).reset_index(drop=True)



                # --- SINGLE FIELD CONDITIONAL

                elif cont_key == 'nonzero':
                    flds = expand_fieldnames( fldsets, cmd_toks )
                    for src_fld in flds:
                        new_fld = src_fld
                        df.loc[ df[src_fld] <= 0.0 , new_fld ] = 0
                        df.loc[ df[src_fld] > 0.0 , new_fld ] = 1


                elif cont_key == 'nanz':
                    flds = expand_fieldnames( fldsets, cmd_toks )
                    for src_fld in flds:
                        magic_num = -9999
                        df[ src_fld ] = df[ src_fld ].fillna(magic_num)   # replace nans with magic
                        df.loc[ df[src_fld] != magic_num , src_fld ] = 1  # replace rest with 1
                        df.loc[ df[src_fld] != 1 , src_fld ] = 0  # replace rest with 0

                    # df.loc[df[src_fld] >= 0.0, new_fld] = 1

                # --- MULTIFIELD CONDITIONAL

                elif cont_key == 'match_2f':
                    dest_fld = cmd_toks[0]
                    src_fld1 = cmd_toks[1]
                    src_fld2 = cmd_toks[2]

                    df[ dest_fld ] = 0
                    df.loc[ df[src_fld1] == df[src_fld2] , dest_fld ] = 1


                # --- below here: affected by order

                elif cont_key == 'shift':
                    offset = int( cmd_toks.pop(0) )
                    if offset >=0:
                        offset_label = 'p'+ str(  abs( offset  ) )
                    else:
                        offset_label = 'n'+ str(  abs( offset  )  )

                    flds = expand_fieldnames( fldsets, cmd_toks )
                    for src_fld in flds:
                        new_fld = src_fld+'_sh'+offset_label
                        df[ new_fld ] = df[src_fld].shift( offset, fill_value=0.0 )

                elif cont_key == 'roll':
                    # TODO: handle edge of trajectory case
                    # consider looping through trajectories and applying this to each section.
                    operator = cmd_toks.pop(0)
                    offset = int(cmd_toks.pop(0) )

                    if offset >=0:
                        offset_label = 'p'+ str(  abs( offset  ) )
                    else:
                        offset_label = 'n'+ str(  abs( offset  )  )

                    flds = expand_fieldnames( fldsets, cmd_toks )
                    for src_fld in flds:
                        new_fld = src_fld+'_roll_'+ operator+'_'+offset_label
                        df[ new_fld ] = df[src_fld].rolling( offset, min_periods=1 ).sum().astype(np.float32)

                elif cont_key == 'query':
                    query = ' '.join(cmd_toks)
                    print('query',query)
                    df = df.query(query)

                elif cont_key == 'eval':
                    eva = ' '.join(cmd_toks)
                    print('eval',eva)
                    df = pd.eval(eva, target=df)

                else:
                    print('*** [prep_script ERROR] command not found: ',cont_key, cmd_toks)
                    exit(1)

        '''
        TODO:
        - add reverse rolling. reverse rolling max for rewards.
            - in general, stretching rewards back
        
        
        '''

        if debug_prep:
            print('   before columns: ',list(df))

        print(df)
        print(' ** final frame')
        d = self.get_replay_summary( df )
        print(d)

        print(obs_stats)

        # TODO: write_replay_spec_path

        dbg("prep","write_replay_path",write_replay_path)
        df.to_csv(write_replay_path, index=False)



    def evaluate_eps( self, vec_env, agent_label, telepath_agent, eval_eps=10, status="", return_raw_scores=True ):
        set_=self.setspec.spec['content']

        mlflow_folder=None
        if 'mlflow_folder' in set_:
            mlflow_folder = set_['mlflow_folder']
        print('eval_eps',eval_eps)
        res=evaluate_eps( vec_env, agent_label, telepath_agent, eval_eps=eval_eps, return_raw_scores=return_raw_scores, mlflow_folder = mlflow_folder  );
        print(agent_label, status, res)
        return res


    def main_train_offline( self, spec_vec_env, replay_path, replay_spec_path, algo_spec, agent_path, eval_env_sim = None ):

        dbg("main_train_offline", comment="load replay", data=replay_path)
        #self.sett['read_replay'].df = storage.read_df_from_location(self.sett['replay_loc'])
        #self.sett['read_replay'].read_from_location(self.sett['replay_loc'])

        # create replay buffer
        common_replay = PdPlainBuffer(replay_path)  # why two?
        common_replay.set_spec( replay_spec_path )


        if ENABLE_WAND:
            wandb.init(sync_tensorboard=True, project="tel3-" + self.conf['label'], dir=self.conf['folder']+'/')

        algo_=algo_spec.spec['content']
        set_=self.setspec.spec['content']

        if eval_env_sim:
            eval_callback = EvalCallback(
                eval_env_sim,
                # callback_on_new_best=callback_on_best,
                #best_model_save_path='agent_best_save',
                log_path=set_['tblog_folder'],  # this is NOT s3 compatible, and Im not sure how to make it so
                eval_freq=1000,
                n_eval_episodes=1,
            )
        else:
            eval_callback = None

        telepath_agent = TelepathAgent_DiscretaAction_Offpolicy_Sbl3()
        telepath_agent._init_agent( algo_, set_, spec_vec_env, )

        # install replay buffer
        telepath_agent.replace_buffer( common_replay )

        if eval_env_sim:
            res=self.evaluate_eps( eval_env_sim, 'telepath_agent batch', telepath_agent, )
            print("telepath_agent batch before learn", res)

        trainsteps=0
        trainsteps_add = algo_['train_steps_per_agent']

        tb_log_name = 's_bdqn_' + telepath_agent.getid()

        if MLFLOW:
            mlflow.create_experiment(tb_log_name,set_['mlflow_folder'])
            #mlflow.set_tracking_uri(set_['mlflow_folder'])


        dbg("main_train_offline", comment="learn_batch", )
        telepath_agent.learn_batch(total_timesteps=trainsteps_add,
                                tb_log_name= tb_log_name,
                                #callback=eval_callback
                                )

        dbg("main_train_offline", comment="telepath_agent.get_buffer_pos", data= telepath_agent.get_buffer_pos() )

        if eval_env_sim:
            res = self.evaluate_eps( eval_env_sim, 'telepath_agent',telepath_agent,  );
            dbg("main_train_offline", comment="telepath_agent after learn", data= res )

        trainsteps += trainsteps_add
        dbg("main_train_offline", comment="trainsteps", data= trainsteps )

        if 'AGENTID' in agent_path:
            agent_path = agent_path.replace('AGENTID',str(telepath_agent.getid()))
        telepath_agent.save_agent( agent_path )



    '''
    Concept: partial buffer
    
    training buffer: action, rewards, obs, next_obs
    policy buffer: (sampleid), action, [obs]
    query buffer: (sampleid), obs
    
    '''

    def _show_explain(self, type, max_depth, explain_col, src_cols, feature_names, df_in, explain_type):

        print('\n[Explanation]')

        for depth in range(1, max_depth+1):
            print(" Depth ", depth)

            if explain_type=='discrete':
                # see https://mljar.com/blog/visualize-decision-tree/ for other options
                clf = DecisionTreeClassifier(random_state=1234, max_depth=depth, criterion='entropy')
            elif explain_type=='continuous':
            # see https://mljar.com/blog/visualize-decision-tree/ for other options
                clf = DecisionTreeRegressor(random_state=1234, max_depth=depth, )

            model = clf.fit(df_in[src_cols], df_in[explain_col])
            text_representation = sklearn.tree.export_text(clf, feature_names=feature_names, show_weights=True)

            y_pred = clf.predict( df_in[src_cols] )
            print(text_representation)

            if type=='class':
                print(classification_report(df_in[explain_col], y_pred, )) #target_names=target_names))


    @timer
    def main_explain_diff(self, replay=None, replay_spec=None, col1=None, col2=None, max_depth=9, explain_type='continuous' ):

        buf=PdPlainBuffer(replay)
        buf.set_spec(replay_spec)
        # in_loc_spec_path
        spec_=buf.specs.spec

        obs_cols =  spec_['fieldmap']['obs_cols']
        #print("obs_cols",obs_cols)

        tid_fld = None
        if 'tid' in spec_['fieldmap']:
            tid_fld = spec_['fieldmap']['tid']
        sid_fld = None
        if 'sid' in spec_['fieldmap']:
            sid_fld = spec_['fieldmap']['sid']
        step_fld = None
        if 'step' in spec_['fieldmap']:
            step_fld = spec_['fieldmap']['step']

        df_in = pd.read_csv(replay)
        #print("df_in all cols",list(df_in))
        #print("Sample")
        #print(df_in[obs_cols].head(3))

        # create comparison columns
        df_in['delta_col']=df_in[col1] - df_in[col2]
        explain_col = 'delta_col'
        print('\n[Compare]')
        print(' delta_col = {} - {} '.format(col1,col2))

        src_cols = obs_cols
        if explain_col in src_cols:
            src_cols.remove(explain_col)
        print("\nExplaining:", explain_col)
        print("Using:", src_cols)

        bins = 10
        hist_width =30

        # show comparisons
        if explain_type == 'discrete':
            # import pdb; pdb.set_trace()
            if 0: pass #sid_fld: value_fld = sid_fld
            elif tid_fld: value_fld = tid_fld
            elif step_fld: value_fld = step_fld

            print('\n[Class comparison] Pivot counts')
            print( df_in.pivot_table(index=col1, columns=col2, values=value_fld, aggfunc='count'))
            print('\n[Class comparison] Pivot ratio')
            total = df_in.shape[0]
            print( df_in.pivot_table(index=col1, columns=col2, values=value_fld, aggfunc='count') / total)

        elif explain_type == 'continuous':
            print('\n[Delta histogram]')
            print(plotille.hist(df_in[explain_col], bins=bins, width=hist_width))

        self._show_explain( type, max_depth, explain_col, src_cols, obs_cols, df_in, explain_type)



    @timer
    def main_explain(self, in_loc=None, in_spec_path=None, explain_col=None, max_depth=9, explain_type='continuous' ):

        buf=PdPlainBuffer(in_loc)
        buf.set_spec(in_spec_path)
        spec_=buf.specs.spec

        obs_cols =  spec_['fieldmap']['obs_cols']
        print("obs_cols",obs_cols)

        tid_fld = None
        if 'tid' in spec_['fieldmap']:
            tid_fld = spec_['fieldmap']['tid']
        sid_fld = None
        if 'sid' in spec_['fieldmap']:
            sid_fld = spec_['fieldmap']['sid']
        step_fld = None
        if 'step' in spec_['fieldmap']:
            step_fld = spec_['fieldmap']['step']

        df_in = pd.read_csv(in_loc)
        print("df_in all cols",list(df_in))
        print("df_in[obs_cols]",df_in[obs_cols].head(3))

        src_cols = obs_cols
        if explain_col in src_cols:
            src_cols.remove(explain_col)
        print("Explaining:", explain_col)
        print("Using:", src_cols)

        self._show_explain( type, max_depth, explain_col, src_cols, obs_cols, df_in, explain_type)




    @timer
    def main_predict(self, agent_map, in_loc=None, in_spec_path=None, out_loc=None, include_obs=False, include_next=False, include_r=False, include_q=False, explain=False, limit=None ):
        # load agent
        #in_loc = 'simplebandit/predict/predict_in.csv'
        #out_loc='predictions.csv'

        dbg("main_predict",data=[ agent_map, in_loc, in_spec_path, out_loc ] )

        buf=PdPlainBuffer(in_loc)
        buf.set_spec(in_spec_path)

        df_in = pd.read_csv(in_loc)
        # TODO: make this work for:
        # - huge files
        # - s3 files
        # - "replay" files with specs

        #df_in = storage.read_df_from_location(in_loc) # pd.read_csv(in_loc)

        #print("TODO: unnormalize prediction features...")
        # TODO: unnormalize prediction features...

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
        print('\n[Replay sample]')
        print(df_in.head(1).T)

        df_out = pd.DataFrame()
        set_=self.setspec.spec['content']


        #def evaluate_eps(env, agent_label, agent, eval_eps=10, deterministic=True, ):
        if MLFLOW:
            #mlflow.set_tracking_uri(set_['mlflow_folder'])
            mlflow.set_experiment('PolicyEval',set_['mlflow_folder'])
            #with mlflow.start_run(run_name=agent_label, ) as run:

        start_index = 0
        batch_size = 1  # only batch size 1 works for now
        start = start_index


        # in_loc_spec_path
        spec_=buf.specs.spec

        obs_cols =  spec_['fieldmap']['obs_cols']
        next_cols = spec_['fieldmap']['next_obs_cols']

        tid_fld = None
        if 'tid' in spec_['fieldmap']:
            tid_fld = spec_['fieldmap']['tid']
        sid_fld = None
        if 'sid' in spec_['fieldmap']:
            sid_fld = spec_['fieldmap']['sid']
        step_fld = None
        if 'step' in spec_['fieldmap']:
            step_fld = spec_['fieldmap']['step']
        if 'reward' in spec_['fieldmap']:
            r_field = spec_['fieldmap']['reward']

        deterministic = True

        iters = 0

        rows=[]

        while True:
            s_time = time.time()

            # determine sample window
            end = start + batch_size

            # went past the end?
            if end > df_in.shape[0] or (limit and start>=limit):
                #print("done")
                break

            # print(" sample " ,start,end)
            s1_time = time.time()
            sample = df_in.iloc[start:end]
            # todo: handle actual batches
            e1_time = time.time()
            d1_time = e1_time - s1_time

            start += batch_size

            # load sample into pytorch

            #sid = genutil.get_rand_sampleid()
            obslist= [  th.tensor(sample[oc].tolist(),dtype=th.float)  for oc in obs_cols ]
            obs = th.stack( obslist, dim=1,)

            row_dict = {}
            if 'sid' in list(sample):
                sid_val = sample.iloc[0][sid_fld]
                row_dict['sid'] = sid_val

            if tid_fld:
                tid_val = sample.iloc[0][tid_fld]
                row_dict[tid_fld] = tid_val

            if step_fld:
                step_val = sample.iloc[0][step_fld]
                row_dict[step_fld] = step_val

            if include_obs:
                for ob in obs_cols:
                    ob_val = sample.iloc[0][ob]
                    row_dict[ob] = ob_val

            if include_next:
                for ob in next_cols:
                    ob_val = sample.iloc[0][ob]
                    row_dict[ob] = ob_val

            if include_r:
                r_val = sample.iloc[0][r_field]
                row_dict[r_field] = r_val

            # agent prediction
            for agent_key in agent_map.keys():
                #action, _states = agent_map[agent_key].predict(obs, deterministic=deterministic)
                action, q_values, _state = agent_map[agent_key].predict_raw(obs, )

                row_dict[agent_key+'_action'] = int(action[0])

                if include_q:
                    #import pdb;pdb.set_trace()
                    for idx,qval in enumerate(q_values[0].tolist()):
                        row_dict[agent_key+'_q'+str(idx)] = q_values[0][idx].cpu().item()

                    for idx, qval in enumerate(q_values[0].tolist()):
                        if idx>0:
                            row_dict[agent_key+'_dq_'+str(idx)+'_0'] = q_values[0][idx].cpu().item() - q_values[0][0].cpu().item()


            # prediction storage

            # if append_obs:
            #
            # row_dict[obs_col]= obslist[idx]

            rows.append( row_dict )

            # print(action)
            iters+= 1
            if iters %5000==0:
                print(datetime.datetime.now(), iters)

        # we are building the pandas df row by row, BAD
        df_out=pd.DataFrame( rows )


        if tid_fld:
            df_out[tid_fld]=df_out[tid_fld].astype(int)
        if step_fld:
            df_out[step_fld]=df_out[step_fld].astype(int)
        #if sid_fld:
        #    df_out[sid_fld]=df_out[sid_fld]

        for agent_key in agent_map.keys():
            df_out[agent_key+'_action']=df_out[agent_key+'_action'].astype(int)

        print('\n[Agents]',)
        print(agent_map)
        for agent_key in agent_map.keys():
            print('=== Agent: '+agent_key,'===================================================')

            print('\n[Action distribution]')
            print( df_out[agent_key+'_action'].value_counts() )

            #print( agent_key )
            #import pdb; pdb.set_trace()

            if include_q:

                print('\n[Q summary]')
                df_qstats = pd.DataFrame([df_out[agent_key + '_q0'].describe(), df_out[agent_key + '_q1'].describe(),
                                          df_out[agent_key + '_dq_1_0'].describe()         ])
                print(df_qstats.T)

                print('\n[Q distributions]')
                bins = 10
                hist_width = 30
                print('\nQ_0')
                print(plotille.hist(df_out[agent_key + '_q0'], bins=bins, width=hist_width))
                print('\nQ_1')
                print(plotille.hist(df_out[agent_key + '_q1'], bins=bins, width=hist_width))

                print('\ndelta_q: Q_1 - Q_0')
                print(plotille.hist(df_out[agent_key + '_dq_1_0'], bins=bins, width=hist_width))

            if explain:
                max_depth=4
                src_cols = obs_cols
                explain_col = agent_key+'_action'
                explain_type = 'continuous'
                self._show_explain(type, max_depth, explain_col, src_cols, obs_cols, df_out, explain_type)
                '''
                print('\n[Explanations]')

                for depth in range(1,9):
                    print(" Depth ",depth)
                    # see https://mljar.com/blog/visualize-decision-tree/ for other options
                    clf = DecisionTreeClassifier(random_state=1234, max_depth=depth, criterion='entropy')

                    explain_col = agent_key+'_action'
                    if limit:
                        model = clf.fit(df_in[obs_cols].head(limit), df_out[explain_col].head(limit) )
                    else:
                        model = clf.fit( df_in[obs_cols], df_out[explain_col])
                    text_representation = sklearn.tree.export_text(clf,feature_names = obs_cols,show_weights=True )
                    print(obs_cols)
                    print(text_representation)
                '''

        print('\n[Output sample]')
        print(df_out.head(1).T)

        sorts = []
        if tid_fld:
            sorts.append(tid_fld)
        if step_fld:
            sorts.append(step_fld)
        #if sid_fld:
        #    sorts.append(sid_fld)

        if sorts:
            df_out = df_out.sort_values(by=sorts)
        df_out.to_csv(out_loc,index=False)




def proc_exec( bash_list, capture_output=True ):
    if capture_output:
        proc = subprocess.run( bash_list, capture_output=capture_output )
    else:
        proc = subprocess.run( bash_list, capture_output=capture_output, stderr=sys.stderr, stdout=sys.stdout)
    return(proc.stdout.decode('ascii'))



def runlogger(method):
    @functools.wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        method_output = method(self, *method_args, **method_kwargs)
        if self.runlog_fn is not None:
            print( "[runlog] writing to", self.runlog_fn)
            data = { 'cmd_str': " ".join(sys.argv), 'cmd_array': sys.argv, }
            storage.write_binary_file(self.runlog_fn , json.dumps(data)  )
        return method_output
    return _impl


class TelepathCmd():

    #te = None

    def __init__(self, project_spec=None ):
        self.te = TelepathEngine()
        if project_spec:
            self.te.telepath_init(project_spec)
        #self.status()
        self._runlogger_start()

    def _runlogger_start( self, ):
        if len(sys.argv)<2:
            return
        cmd = sys.argv[1]
        cont = self.te.setspec.spec['content']
        if 'runlog_folder' not in cont:
            self.runlog_fn = None
            return
        self.runlog_folder = cont['runlog_folder']
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.runlog_fn = cont['runlog_folder']
        if self.runlog_fn[-1]!='/':
            self.runlog_fn += '/'
        xid = self.te.execution_id
        self.runlog_fn += cmd+'_'+ts+'_'+xid+'.json'



    @runlogger
    def replay_stats(self, replay, col=None ):
        '''
        Show statistics on replay file --replay=... ; optionally limit to column --col=..
        '''
        self.te = TelepathEngine()
        print(replay)
        df = pd.read_csv(replay,)
        if col:
            bins=10
            hist_width=30
            print(plotille.hist(df[col], bins=bins, width=hist_width))
        else:
            pp(sorted(list(df)))
            sd=self.te.get_replay_summary( df )
            print(sd)


    def _replay_perf_stats(self, replay):
        read_replay = PdPlainBuffer(replay)
        print( read_replay )
        bf=read_replay._buffer_perf_stats()
        print(bf)

    @runlogger
    def train_online(self, env_id, replay_loc, replay_spec ): # agent_dest, train_config): # agent_dest, env_src, train_online params
        '''
        in: env spec
        out: replay buffer ; trained online agent
        '''
        vec_env = get_vec_env_from_label( env_id )
        self.te.main_generate_replay( vec_env, replay_loc, replay_spec )


    @runlogger
    def generate_replay(self, env_id, out_replay, replay_spec, algo_spec, agent_dest ): # agent_src, env_src, rollout params, replay_location
        '''
        in: env spec
        out: replay buffer ; trained online agent
        '''
        algo_spec_path = algo_spec
        vec_env_sim = get_vec_env_from_label(env_id)
        algo_sp = AlgoSpec()
        algo_sp.load(algo_spec_path)

        self.te.main_generate_replay( vec_env_sim=vec_env_sim, write_replay_path=out_replay, write_replay_spec_path=replay_spec, algo_spec=algo_sp, online_agent_dest_path=agent_dest)


    @runlogger
    def prep(self,  in_replay, out_replay, prep_spec, prep_mode, debug=False  ):
        '''
        Prepare a raw replay file using a prep_spec, to produce an prepared replay file (for train or predict, depending on prep_mode).
        '''
        return self.te.main_prepare_replay(  in_replay, out_replay, prep_spec, prep_mode, debug_prep=debug  )

    @runlogger
    def train(self, replay_spec, replay, algo_spec, agent_path, eval_env=None ): # replay_src, env_id, rollout params, agent_dest
        '''
        Train agent on replay data
        '''

        algo_sp = AlgoSpec()
        algo_sp.load(algo_spec)

        replay_spec_path = replay_spec
        rspec = ReplaySpec()
        rspec.load(replay_spec_path)
        spec_gym = get_specgym_from_replay_spec( rspec )
        spec_vec_env = get_vectorize_env( spec_gym )

        if eval_env:
            eval_env_sim = get_vec_env_from_label( eval_env )
        else:
            eval_env_sim = None

        self.te.main_train_offline( spec_vec_env = spec_vec_env, replay_path = replay, algo_spec = algo_sp, replay_spec_path=replay_spec_path, agent_path = agent_path, eval_env_sim=eval_env_sim )


    @runlogger
    def eval_gym(self, agent_path, env_id, eval_eps=10 ):
        '''
        Evaluate an agent on a gym
        '''
        telepath_agent = TelepathAgent_DiscretaAction_Offpolicy_Sbl3()
        telepath_agent.load_agent( agent_path )
        print(telepath_agent)
        print(telepath_agent._agent)
        vec_env_sim = get_vec_env_from_label( env_id )
        res=self.te.evaluate_eps( vec_env_sim, 'telepath_agent', telepath_agent, status="batch_eval", eval_eps=eval_eps, return_raw_scores=True);
        pp(res)
        bins=20
        hist_width=None
        print( plotille.hist(res['scores'], bins=bins, width=40) )


    @runlogger
    def predict(self, agent_path, in_features_path, in_spec, out_pred_path, include_obs=False, include_next=False, include_q=False, include_r=False, explain=False, limit=None ):
        '''
        Use agent on input data, write predictions to a file
        '''
        telepath_agents_batch=[]

        if ',' in agent_path:
            agent_paths = agent_path.split(",")
            agent_keys = [ 'agent'+str(i) for i in range(0,len(agent_paths)) ]
        else:
            agent_paths = [ agent_path ]
            agent_keys = ['batch_agent']

        agent_map = {}
        for key, path in zip( agent_keys, agent_paths):
            telepath_agent_batch = TelepathAgent_DiscretaAction_Offpolicy_Sbl3()
            telepath_agent_batch.load_agent( path )
            telepath_agents_batch.append(telepath_agent_batch)
            agent_map[key] = telepath_agent_batch

        self.te.main_predict( agent_map=agent_map, in_loc=in_features_path, in_spec_path=in_spec, out_loc=out_pred_path, include_obs=include_obs, include_next=include_next, include_q=include_q, include_r=include_r, explain=explain, limit=limit )


    @runlogger
    def explain_diff(self, replay, replay_spec, col1, col2, max_depth=9, explain_type='continuous' ):
        '''
        Use agent on input data, write predictions to a file
        '''
        self.te.main_explain_diff(  replay=replay, replay_spec=replay_spec, col1=col1, col2=col2, max_depth=max_depth, explain_type=explain_type )


    @runlogger
    def explain(self, replay, replay_spec, explain_col='batch_agent_action', max_depth=9, explain_type='continuous' ):
        '''
        Use agent on input data, write predictions to a file
        '''
        self.te.main_explain(  in_loc=replay, in_spec_path=replay_spec, explain_col=explain_col, max_depth=max_depth, explain_type=explain_type )


    def tb(self):
        '''
        Run tensorboard view training curves
        '''
        tb_log = self.te.setspec.spec['content']['tblog_folder']
        print("Running tensorboard, visit port 6006")
        print( proc_exec(['tensorboard','--logdir='+tb_log,'--bind_all' ], capture_output=False,) )
        subprocess.run(['ls'], stderr=sys.stderr, stdout=sys.stdout)


    def info(self):
        '''
        Version info
        '''
        print("Telepath version " + telepath_version)
        print("Copyright (c) 2020 Pathway Intelligence Inc.")



    def status(self):
        '''
        System status
        '''
        self.info()
        print("")
        print('torch.__version__',th.__version__)
        print('torch.version.cuda',th.version.cuda)
        print("")
        print( proc_exec(['cat','/etc/issue']) )
        print( proc_exec(['uname','-a']) )
        print( proc_exec(['free','-h']) )
        print( proc_exec(['lscpu']) )
        print("")
        print( proc_exec(['nvidia-smi']) )
        pp(self.te.setspec.spec)



te = TelepathEngine()


if __name__ == '__main__':
    fire.Fire(TelepathCmd)
    exit(0)



'''
Use cases: local
---------

# generate_replay
python telepath.py generate_replay --config=specs2/telepath_settings.yaml --env_id=CartPole-v0 --replay_path=/tmp/replay.csv --replay_spec=specs2/spec_replay_cartpole_tiny.yaml   --algo_spec=specs2/tiny_spec_algo.yaml --agent_dest=/tmp/agent.online.telepath

#train
python telepath.py train --config=specs2/telepath_settings.yaml --env_id=CartPole-v0 --replay_path=/tmp/replay.csv --replay_spec=specs2/spec_replay_cartpole_tiny.yaml   --algo_spec=specs2/tiny_spec_algo.yaml --agent_dest=/tmp/agent.online.telepath

#   todo: remove CartPole ident; read this data from the replay file

# predict
python telepath.py predict --config=specs2/telepath_settings.yaml --in_features_path=/tmp/replay.csv --in_spec=specs2/spec_replay_cartpole_tiny.yaml    --agent_path=/tmp/agent.online.telepath --out_pred_path=/tmp/predict.csv

Use cases: factored
---------

proj_spec=specs2/telepath_settings.yaml
replay_path=/tmp/replay.csv
replay_spec=specs2/spec_replay_cartpole_tiny.yaml
algo_spec=specs2/tiny_spec_algo.yaml
agent_path=/tmp/agent.batch.telepath
online_agent_path=/tmp/agent.online.telepath
out_pred_path=/tmp/predict.csv

# generate_replay
python telepath.py generate_replay --config=$proj_spec --env_id=CartPole-v0 --replay_path=$replay_path --replay_spec=$replay_spec   --algo_spec=$algo_spec --agent_dest=$online_agent_path

#train
python telepath.py train --config=$proj_spec --env_id=CartPole-v0 --replay_path==$replay_path --replay_spec=$replay_spec   --algo_spec=$algo_spec --agent_dest=$agent_path

#   todo: remove CartPole ident; read this data from the replay file

# predict
python telepath.py predict --config=$proj_spec --in_features_path=$replay_path --in_spec=$replay_spec    --agent_path=$agent_path --out_pred_path=$out_pred_path





S3
----

# generate_replay
python telepath.py generate_replay --config=specs2/telepath_settings.yaml --env_id=CartPole-v0 --replay_path=/tmp/replay.csv --replay_spec=specs2/spec_replay_cartpole_tiny.yaml   --algo_spec=specs2/tiny_spec_algo.yaml --agent_dest=/tmp/agent.online.telepath

# train
python telepath.py train --config=specs2/telepath_settings.yaml --env_id=CartPole-v0 --replay_path=/tmp/replay.csv --replay_spec=specs2/spec_replay_cartpole_tiny.yaml   --algo_spec=specs2/tiny_spec_algo.yaml --agent_dest=/tmp/agent.online.telepath

# todo: remove CartPole ident; read this data from the replay file

# predict
python telepath.py predict --config=specs2/telepath_settings.yaml --in_features_path=/tmp/replay.csv --in_spec=specs2/spec_replay_cartpole_tiny.yaml    --agent_path=/tmp/agent.online.telepath --out_pred_path=/tmp/predict.csv



ALSO:
- get gym spec from replay
- evaluate policy?

'''
