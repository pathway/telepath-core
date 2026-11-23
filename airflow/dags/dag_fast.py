from datetime import timedelta

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG
# Operators; we need this to operate!
from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago
# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(0),
    'email': ['robin@pathwayi.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(seconds=2),
    # 'queue': 'bash_queue',
    #'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}

env = {
    'PATH': '/home/user/miniconda3/envs/telepath-core/bin:/home/user/miniconda3/bin:/usr/local/cuda/bin:/home/user/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/user/.local/bin:/usr/lib/jvm/default-java/bin:/usr/lib/jvm/default-java/jre/bin',
    'PWD': '/path/to/telepath-core/telepath_worker/',
    'HOME': '/home/user/',
    #'PYTHONPATH': '/opt/example/bin/',
    #'DJANGO_SETTINGS_MODULE' : 'agbase_admin.settings_prod',
}


dag = DAG(
    'dag_fast',
    default_args=default_args,
    description='dag_fast',
    schedule_interval=timedelta(minutes=60),
)

def get_bash_worker(s):
    return 'source /home/user/miniconda3/etc/profile.d/conda.sh && conda activate telepath-core && cd /path/to/telepath-core/telepath_worker/ && ' + s + ' ' # space at end of string required for obscure bug

t1 = BashOperator(
    task_id='pytest_perf_agent',
    bash_command=get_bash_worker('pytest test_agent_perf/'),
    env=env,
    dag=dag,
)

t2 = BashOperator(
    task_id='integ_base_offline',
    depends_on_past=False,
    bash_command=get_bash_worker('python integ_base_offline.py'),
    env=env,
    dag=dag,
)

t3 = BashOperator(
    task_id='pytest_test',
    depends_on_past=False,
    bash_command=get_bash_worker('pytest test/'),
    env=env,
    dag=dag,
)

#t1 >> t2
