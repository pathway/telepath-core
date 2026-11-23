from telepath_worker.integ_test import integration_lib
from pprint import pprint as ppp
import fire

'''
This int test allows:
- running via docker vs running locally

'''

def run_integ( use_docker = False, show_cmds_only=False, docker_version=None ):
    if docker_version is None:
        docker_version= integration_lib.docker_version

    project_label='lunar_small'
    spec_relative= 'specs/'+project_label
    #project_root='s3://client-telepath-staging/test1'    # no trailing slash
    #project_root='s3://example-telepath-bucket/client_sampledata_nov21_2020'    # no trailing slash
    #project_root='s3://example-telepath-test-bucket/test1'    # no trailing slash

    if use_docker:
        project_root = 's3://example-telepath-test-bucket/cartpole_small/'
        base_local = '/src/telepath-core/telepath_worker/'
    else:
        project_root = '/path/to/telepath-core/projects/'+project_label+'/'
        base_local = ''

    spec_root=base_local + spec_relative  # makes sense for docker only

    params = integration_lib.get_params(spec_root, project_root)
    params['env_id'] = 'LunarLander-v2'
    #params['env_id'] = 'SimpleGym_Sequential_Short-v0'

    scripts = integration_lib.get_scripts(params)

    #result = integration_lib.exec_script(  scripts['fulltest_offline_prep'], exec_prefix=exec_prefix, show_cmds_only=show_cmds_only )
    result = integration_lib.exec_script_gen(  scripts['fulltest_offline_prep'], docker_version=docker_version, use_docker=use_docker, show_cmds_only=show_cmds_only )

    exit(result)

fire.Fire(run_integ)

