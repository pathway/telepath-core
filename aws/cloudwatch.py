import boto3
import datetime
import time
import pandas as pd

pd.set_option('max_columns', None)
pd.set_option("max_rows", None)
pd.set_option("max_colwidth", None)
pd.set_option('display.width', 1000)


client = boto3.client('logs')

query = "fields @timestamp, @message | sort @timestamp desc | limit 200"
#          "# | parse @message \"username: * ClinicID: * nodename: *\" as username, ClinicID, nodename | filter ClinicID = 7667 and username='user+test@example.com'"

log_group = '/aws/batch/job'

print( time.time() )
print( time.gmtime() )
print( time.mktime ( time.gmtime()) )

start_query_response = client.start_query(
    logGroupName=log_group,
    startTime=int(  time.mktime(time.gmtime()) - 1000*3600*0.1  ),
    endTime=int( time.mktime( time.gmtime()) ),
    queryString=query,
)

query_id = start_query_response['queryId']

response = None
rows = []

while response == None or response['status'] == 'Running':
    print('Waiting for query to complete ...')
    time.sleep(1)
    response = client.get_query_results(
        queryId=query_id
    )
    res = response['results']
    row = {}
    for r in res:
        #print(r)
        row = { r[0]['field']: r[0]['value'],
                r[1]['field']: r[1]['value'],
                #r[2]['field']: r[2]['value'],
                }
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df)
    #print(response)
