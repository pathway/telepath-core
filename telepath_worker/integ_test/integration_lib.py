import subprocess
import sys
from pprint import pprint as pp

docker_version = '0.8.1.15'

def proc_exec( bash_list, capture_output=True ):
    if capture_output:
        proc = subprocess.run( bash_list, capture_output=capture_output )
        return(proc.stdout.decode('ascii'))
    else:
        proc = subprocess.run( bash_list, capture_output=capture_output, stderr=sys.stderr, stdout=sys.stdout)
        print('----')
        print("proc_exec result:",proc.returncode, proc)
        return proc.returncode

def x(s):
    print("[executing]",s)
    result = proc_exec(s.split(' '), False)
    #print( result )
    return result

#Implied.
#scripts['fulltest_basic'] =[ 'python telepath.py fulltest --config=config/cartpole_tiny.yaml',
#'head /path/to/telepath-runs/cartpole/replay/full_replay.csv',]

def exec_script(script, exec_prefix='', show_cmds_only=False):
    #print(script)
    for line in script:
        if show_cmds_only:
            print('#')
            if exec_prefix:
                print(exec_prefix +' '+ line)
            else:
                print(line)
            continue
        print("")
        print('-- exec_script ' +'-'*50)
        print(exec_prefix, line)
        print('-'*70)

        result=x(exec_prefix  + line)
        if result!=0:
            print("*** ERROR ***")
            return result
    return 0

def exec_script_gen( script, use_docker=False, docker_version=None, show_cmds_only=False):
    if use_docker:
        exec_prefix = 'docker container run --gpus all -it telepath:{version} '.format(version=docker_version)
    else:
        exec_prefix = ''

    result = exec_script(script, exec_prefix=exec_prefix, show_cmds_only=show_cmds_only)
    return result


def exec_all( script_dict, exec_prefix='', show_cmds_only=False ):
    if not show_cmds_only:
        print("script_dict")
        pp(script_dict)

    for k in script_dict.keys():

        if not show_cmds_only:
            print('*** exec_all **' + '*'*80)
            print(k,)
            print('*'*100)
        result = exec_script( script_dict[k], exec_prefix, show_cmds_only )
        if result!= 0:
            return result
    return 0


def get_scripts(params):
    scripts={}

    scripts['line_gen'] =  "./telepath generate_replay --project_spec={proj_spec} --env_id={env_id} --out_replay={rawtrain_path} --replay_spec={raw_replay_spec} --algo_spec={algo_spec} --agent_dest={online_agent_path}".format(**params)

    scripts['line_prep_train'] =  "./telepath prep --project_spec={proj_spec} --in_replay={rawtrain_path} --out_replay={replay_train_path} --prep_spec={prep_spec} --prep_mode=train".format(**params)

    scripts['line_prep_predict'] =  "./telepath prep --project_spec={proj_spec} --in_replay={rawtrain_path} --out_replay={replay_predict_path} --prep_spec={prep_spec} --prep_mode=predict ".format(**params)

    scripts['line_train'] =  "./telepath train --project_spec={proj_spec} --replay={replay_train_path} --replay_spec={replay_spec} --algo_spec={algo_spec} --agent_path={agent_path}".format(**params)

    scripts['line_predict'] =  "./telepath predict --project_spec={proj_spec} --in_features_path={replay_predict_path} --in_spec={replay_spec} --agent_path={agent_path} --out_pred_path={out_pred_path} --include_q --include_obs".format(**params)  #

    scripts['line_predict_train'] =  "./telepath predict --project_spec={proj_spec} --in_features_path={replay_train_path} --in_spec={replay_spec} --agent_path={agent_path} --out_pred_path={out_pred_path} --include_q --include_obs ".format(**params) #  --explain --include_q --include_obs


    scripts['line_predict_explain1'] =  "./telepath explain --project_spec={proj_spec} --replay={out_pred_path} --replay_spec={replay_spec} --explain_col=batch_agent_action --explain_type=discrete --max_depth=4".format(**params)  #

    scripts['line_predict_train_explain'] =  "./telepath explain --project_spec={proj_spec} --replay={out_pred_path} --replay_spec={replay_spec} --explain_col=batch_agent_action --explain_type=discrete --max_depth=4".format(**params)  #

    scripts['line_eval_gym'] =  "./telepath eval_gym --project_spec={proj_spec} --agent_path={agent_path} --env_id={env_id} --eval_eps=100".format(**params)  #



    scripts['fulltest_offline_prep']= [
        scripts['line_gen'],
        scripts['line_prep_train'],
        scripts['line_train'],
        scripts['line_prep_predict'],
        scripts['line_predict'],
        scripts['line_predict_explain1'],
        scripts['line_eval_gym']
    ]

    scripts['fulltest_offline_noprep']= [
        scripts['line_gen'],
        scripts['line_train'],
        scripts['line_predict'],
        scripts['line_eval_gym']
    ]

    scripts['offline_prep_train_pred']= [
        scripts['line_prep_train'],
        scripts['line_train'],
        scripts['line_prep_predict'],
        scripts['line_predict'],
        scripts['line_predict_train'],
        scripts['line_predict_train_explain']
    ]

    return scripts


def get_params(spec_root, project_root):
    params = {
        'proj_spec': spec_root + '/spec_project.yaml',
        'raw_replay_spec': spec_root + '/spec_replay.yaml',
        'prep_spec': spec_root + '/spec_replay_prep.yaml',  # not used?
        'replay_spec': spec_root + '/spec_replay.yaml',  # not used
        'algo_spec': spec_root + '/spec_algo.yaml',

        'rawtrain_path': project_root + '/raw_replay.csv',
        'replay_path': project_root + '/replay.csv',
        'replay_predict_path': project_root + '/replay_predict.csv',
        'replay_train_path': project_root + '/replay_train.csv',
        'agent_folder': project_root + '/agent/',
        'online_agent_folder': project_root + '/online.agent_telepath',  # why almost duped?
        'online_agent_path': project_root + '/online.agent_telepath',
        'agent_path': project_root + '/batch.agent_telepath',
        'out_pred_path': project_root + '/predict.csv',
        'env_id': 'None',
    }
    for k in params.keys():
        print("{}: {}".format(k, params[k]))
    return params


if __name__=="__main__":
    res=x("ls")
    print(res)
