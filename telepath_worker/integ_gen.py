#!/usr/bin/env python3

from telepath_worker.integ_test import integration_lib
from pprint import pprint as ppp
import fire

'''
Generate command lines for local or docker usage

'''

#version='0.8.1.10-dev'
#DOCKER = True
#if DOCKER:
#    exec_prefix='docker container run --gpus all -it telepath:{version} '.format(version=version)
#else:
#    exec_prefix=''
#exit(result)

def integ_cmd(spec_root, data_root, env_id, exec_prefix=''):
    print('# spec_root',spec_root)
    print('# data_root',data_root)
    print('# env_id',env_id)
    print('# exec_prefix',exec_prefix)

    params = {
        'proj_spec': spec_root + '/spec_project.yaml',
        'raw_replay_spec': spec_root + '/spec_replay.yaml',
        'prep_spec': spec_root + '/spec_replay_prep.yaml',  # not used?
        'replay_spec': spec_root + '/spec_replay.yaml',  # not used
        'algo_spec': spec_root + '/spec_algo.yaml',

        'rawtrain_path': data_root + '/replay_rawtrain.csv',  # not used
        'replay_predict_path': data_root + '/replay_predict.csv',
        'replay_train_path': data_root + '/replay_train.csv',
        'agent_folder': data_root + '/agent/',
        'online_agent_folder': data_root + '/online.agent_telepath',  # why almost duped?
        'online_agent_path': data_root + '/online.agent_telepath',
        'agent_path': data_root + '/batch.agent_telepath',
        'out_pred_path': data_root + '/predict.csv',

        'env_id':env_id,
    }

    scripts = integration_lib.get_scripts(params)
    result = integration_lib.exec_script( scripts['fulltest_offline_prep'], exec_prefix=exec_prefix, show_cmds_only=True )

if __name__ == '__main__':
    fire.Fire(integ_cmd)
    exit(0)

