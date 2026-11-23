import boto3
import os
import subprocess
import pandas as pd
from pprint import pprint as pp
import fire
import sys
import yaml
from datetime import datetime, timedelta
import time
from botocore.config import Config
import json


conf_file = 'telepath_aws_east.yaml'
conf_file = 'telepath_aws_west3.yaml'
with open(conf_file) as f:
    sett = yaml.load(f, Loader=yaml.FullLoader)
print(sett)


my_config = Config(
    region_name = sett['aws_region'],
)


cmd='nvidia-smi'
#cmd='python3 telepath.py fulltest --config=config/docker_cartpole_s3.yaml'
#cmd='whoami'
# && date && uname -a'
#cmd='nvidia-smi'

action=sys.argv[1]

batch = boto3.client('batch',config=my_config)


pd.set_option('max_columns', None)
pd.set_option("max_rows", None)
pd.set_option("max_colwidth", None)
pd.set_option('display.width', 1000)


def list_jobs_status( jobQueue, status ):
    response = batch.list_jobs(
        jobQueue=jobQueue,
        #arrayJobId='string',
        #multiNodeJobId='string',
        jobStatus= status,
        maxResults=100,
        #nextToken='string'
    )
    #print(response)
    if response['jobSummaryList']:
        return pd.DataFrame(response['jobSummaryList'])
    return None



if action=='launch':
    response = batch.submit_job(jobName=sett['jobName'],
        jobQueue=sett['jobQueue'], # sufficient for most jobs
        jobDefinition=sett['jobDefinition'], # use a real job definition
        containerOverrides={
            "command": cmd.split(' '),
            #"command": ['echo', 'hello', 'world'], # optionally override command
            "environment": []
        })
    jid = response['jobId']
    print("Job ID is {}.".format(jid))

    '''
    Errors
    ------
    PermissionError: [Errno 13] Permission denied: '/src/telepath/mlruns'

    '''

elif action=='list_jobs':
    rets=[]
    for s in ['SUBMITTED','PENDING','RUNNABLE','STARTING','RUNNING','SUCCEEDED','FAILED',]:
        ret = list_jobs_status( sett['jobQueue'], s )
        if ret is not None:
            rets.append(ret)
    if not rets:
        df = pd.DataFrame()
        print("No jobs")
    else:
        df = pd.DataFrame( rets[0] )
        for i in range(1,len(rets)):
            df_sub = pd.DataFrame( rets[i] )
            df = df.append(df_sub)

        df['dt_utc_createdAt'] = pd.to_datetime(df['createdAt'], unit='ms')
        df = df.sort_values(by='dt_utc_createdAt',ascending=False)
    #print(df[['jobId','dt_utc_createdAt','status','statusReason',]].head(12))
        print(df.head(12))
        #print(df.container.head(12))

elif action=='events':

    # Create CloudWatchEvents client
    cloudwatch_events = boto3.client('events',config=my_config)

    # Put an event
    response = cloudwatch_events.put_events(
        Entries=[
            {
                'Detail': json.dumps({'key1': 'value1', 'key2': 'value2'}),
                'DetailType': 'appRequestSubmitted',
                'Resources': [
                    'RESOURCE_ARN',
                ],
                'Source': 'com.company.myapp'
            }
        ]
    )
    print(response['Entries'])



elif action=='validate':
    response = batch.describe_job_queues(
        jobQueues=[ sett['jobQueue'], ],
        #maxResults=10,
        nextToken='string'
    )
    print( pd.DataFrame(response['jobQueues']) )

    response = batch.describe_job_queues(
        jobQueues=[ sett['jobQueue'], ],
        #maxResults=10,
        nextToken='string'
    )
    print( pd.DataFrame(response['jobQueues']).T )


    response = batch.describe_job_definitions(
        jobDefinitions=[
            sett['jobDefinition'],
        ],
        #jobDefinitionName='string',
        #status='string',
        #nextToken='string'
    )
    print( pd.DataFrame(response['jobDefinitions']).T )

    '''
    elif action=='show_api':
        response = batch.describe_jobs(
            jobs = jids
        )
        #print(response)
        df = pd.DataFrame(response['jobs'])
        df['dt_utc'] = pd.to_datetime(df['createdAt'], unit='ms')
        df = df.sort_values(by='dt_utc',ascending=False)
        # should print this:
        #df[ ~df.container.isna() ] ['cmd'] = df['container'][0]['command']
        print( df[['dt_utc','jobId','status','statusReason',]]) #'container']]  )
    
        #print( list(df)  )
        # print( pp(response['jobs'] ) )
    '''

elif action=='kill':
    cmd = 'aws batch describe-jobs --jobs '+str(jid)
    output = subprocess.check_output(cmd, shell=True)
    print(cmd)
    print(output.decode('ascii') )

elif action=='logs':

    client = boto3.client('logs',config=my_config)

    query = "fields @timestamp, @message | sort @timestamp desc | limit 500"
    #          "# | parse @message \"username: * ClinicID: * nodename: *\" as username, ClinicID, nodename | filter ClinicID = 7667 and username='user+test@example.com'"

    log_group = '/aws/batch/job'

    start_query_response = client.start_query(
        logGroupName=log_group,
        startTime=int((datetime.today() - timedelta(hours=24 * 7)).timestamp()),
        endTime=int(datetime.now().timestamp()+1000),
        queryString=query,
    )

    query_id = start_query_response['queryId']

    response = None
    rows = []

    while response == None:  # or response['status'] == 'Running':
        print('Waiting for query to complete ...')
        time.sleep(1)
        response = client.get_query_results(
            queryId=query_id,
        )
        res = response['results']
        row = {}
        for r in res:
            # print(r)
            row = {r[0]['field']: r[0]['value'],
                   r[1]['field']: r[1]['value'],
                   # r[2]['field']: r[2]['value'],
                   }
            rows.append(row)
        df = pd.DataFrame(rows)
        print(df)
        # print(response)

'''
ERRORS

- if I run as nobody:
92  2020-09-12 23:49:14.494                                   PermissionError: [Errno 13] Permission denied: '/nonexistent'

- when I run code that needs GPU:
nvidia-smi not installed
nvidia drivers not found

- if I run as root with priviledges, and GPU, job stays in RUNNABLE state.

'''
