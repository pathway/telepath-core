from telepath_worker import integration_lib
from pprint import pprint as ppp
import os

#project_label='simplegym_seq_short'
#project_root='s3://client-telepath-staging/test1'    # no trailing slash
project_root='s3://example-telepath-bucket/simplegym_test1'    # no trailing slash
project_root='/tmp/simplegym_test1'

if not os.path.exists(project_root):
    os.mkdir(project_root)

base_local = ''

#project_root='s3://example-telepath-test-bucket/test1'    # no trailing slash
spec_root=base_local + 'specs/simplegym_test1'   # makes sense for docker only


params = {
    'proj_spec': spec_root + '/spec_project.yaml',
    'raw_replay_spec': spec_root + '/spec_replay.yaml',
    'prep_spec': spec_root + '/spec_replay_prep.yaml',                # not used?
    'replay_spec': spec_root + '/spec_replay.yaml',  # not used
    'algo_spec': spec_root + '/spec_algo.yaml',

    'rawtrain_path': project_root+'/replay_rawtrain.csv', # not used
    'replay_path': project_root + '/replay.csv',
    'agent_folder': project_root + '/agent/',
    'online_agent_folder': project_root + '/online.agent_telepath', # why almost duped?
    'online_agent_path': project_root + '/online.agent_telepath',
    'agent_path': project_root + '/batch.agent_telepath',
    'out_pred_path': project_root + '/predict.csv',

    'env_id': 'SimpleGym_Sequential_Short-v0',
}

scripts = integration_lib.get_scripts(params)

result = integration_lib.exec_script( scripts['fulltest_offline_prep'], show_cmds_only=False )

exit(result)
