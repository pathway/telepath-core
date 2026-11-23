from telepath_worker.integ_test import integration_lib
from pprint import pprint as ppp
import fire

'''
This int test allows:
- running via docker vs running locally

'''

def run_integ_generic( project_label='cartpole_small', script_name='fulltest_offline_prep', env_id='CartPole-v0',
                       use_docker = False, dryrun=False, docker_version=None, ):

    spec_relative= 'specs/'+project_label
    #project_root='s3://client-telepath-staging/test1'    # no trailing slash
    #project_root='s3://example-telepath-bucket/client_sampledata_nov21_2020'    # no trailing slash
    #project_root='s3://example-telepath-test-bucket/test1'    # no trailing slash

    if use_docker:
        project_root = 's3://example-telepath-test-bucket/'+project_label
        base_local = '/src/telepath-core/telepath_worker/'
    else:
        project_root = '/path/to/telepath-core/projects/'+project_label+'/'
        base_local = ''

    spec_root=base_local + spec_relative  # makes sense for docker only

    params = integration_lib.get_params(spec_root, project_root)
    params['env_id'] = env_id
    #params['env_id'] = 'SimpleGym_Sequential_Short-v0'

    scripts = integration_lib.get_scripts(params)

    #result = integration_lib.exec_script(  scripts['fulltest_offline_prep'], exec_prefix=exec_prefix, show_cmds_only=show_cmds_only )
    result = integration_lib.exec_script_gen(  scripts[script_name], docker_version=docker_version, use_docker=use_docker, show_cmds_only=dryrun )

    exit(result)


projects = {
    'cartpole_small': {'script_name': 'fulltest_offline_prep', 'env_id': 'CartPole-v0', },
    'cartpole_large': {'script_name': 'fulltest_offline_prep', 'env_id': 'CartPole-v0', },
    'lunar_small':    {'script_name': 'fulltest_offline_prep', 'env_id': 'LunarLander-v2', },
    'lunar_large':    {'script_name': 'fulltest_offline_prep', 'env_id': 'LunarLander-v2', },
}

def run_integ( project_label, use_docker = False, dryrun=False, docker_version=None ):
    script_name = projects[ project_label ]['script_name']
    env_id = projects[ project_label ]['env_id']

    return run_integ_generic(project_label=project_label, script_name=script_name, env_id=env_id,
                      use_docker=use_docker, dryrun=dryrun, docker_version=docker_version, )

fire.Fire(run_integ)
