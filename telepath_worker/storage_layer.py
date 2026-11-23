from io import StringIO
import boto3
import io
import pandas as pd
import pickle
from telepath_worker.telepath_utils import dbg
from pathlib import Path
from urllib.parse import urlparse
from slugify import slugify

def write_binary_file( path, data ):
    if is_s3_location(path):
        bucket, key = parse_s3_path(path)
        return write_s3( data, bucket, key )
    else:
        with open(path, 'w') as outfile:
            outfile.write( data )


def parse_s3_path( path ):
    parsed = urlparse(path)
    #print(parsed)
    bucket, key = parsed.netloc, parsed.path[1:]
    return bucket, key

def read_s3(bucket,key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    return io.BytesIO(obj['Body'].read())

def write_s3(data, bucket, key):
    s3_res = boto3.resource("s3")
    dbg('storage.write_data_to_s3', data=[bucket, key])
    s3_res.Object(bucket, key).put(Body=data)


def read_path_from_s3( path ):
    bucket, key = parse_s3_path( path )
    bytes = read_s3(bucket,key)
    return bytes


def read_pickle_from_s3(bucket, key):
    body = read_s3(bucket,key)
    dbg('storage.read_pickle_from_s3', data=[bucket, key] )
    data = pickle.loads(body)
    return data


def ensure_folder_exists(folder):
    '''
    This isnt right.
    '''
    if folder[:2]=='s3':
        return True # ?
    return Path(folder).mkdir(parents=True, exist_ok=True)



def write_df_to_s3(df, bucket, key):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, sep=",", index=False)
    write_s3( data=csv_buffer.getvalue(), bucket=bucket, key=key )


def read_df_from_s3(bucket, key):
    bytes=read_s3(bucket, key)
    dbg('storage.read_df_from_s3', data=[bucket, key] )
    df = pd.read_csv( bytes )
    return df


def is_s3_location(loc):
    # better to parse
    if "://" in loc.lower():
        loc_split=loc.split("://")
        if loc_split[0].lower()=="s3":
            return True
    return False


def parse_s3_location(loc):
    if not is_s3_location(loc):
        return None, None
    loc_split=loc.split("://")
    #print(loc_split)
    bucket_loc=loc_split[1]
    bsplit = bucket_loc.split("/")
    bucket = bsplit[0]
    fn = "/".join ( bsplit[1:] )
    return bucket, fn


def read_pickle_from_location(loc):
    '''
    loc can be s3 location like s3://bucket/filename.csv
    or a regular filepath /usr/local/thing.bin
    '''
    if is_s3_location(loc):
        bucket, key = parse_s3_location(loc)
        dbg('storage.read_pickle_from_location s3', data=[bucket, key])
        data = read_pickle_from_s3(bucket,key)

    #elif type(data)==pd.DataFrame:
    #    data = pd.read_pickle(loc)
    else:
        dbg('storage.read_pickle_from_location file', data=loc)
        pickle_in = open(loc, "rb")
        data = pickle.load(pickle_in)
    return data


def read_df_pickle_from_location(loc):
    '''
    loc can be s3 location like s3://bucket/filename.csv
    or a regular filepath /usr/local/thing.bin
    '''
    if is_s3_location(loc):
        bucket, key = parse_s3_location(loc)
        dbg('storage.read_df_pickle_from_location s3', data=[bucket, key])
        data = read_pickle_from_s3(bucket,key)
    else:
        dbg('storage.read_df_pickle_from_location s3', data=loc)
        data = pd.read_pickle(loc)

    return data




def write_pickle_to_location(data,loc,):
    '''
    loc can be s3 location like s3://bucket/filename.csv
    or a regular filepath /usr/local/thing.bin
    '''
    if is_s3_location(loc):
        bucket, key = parse_s3_location(loc)
        pkl_obj = pickle.dumps(data)
        dbg('storage.write_pickle_to_location: s3',data=[bucket,key])
        write_s3( pkl_obj,bucket,key,)

    elif type(data)==pd.DataFrame:
        dbg('storage.write_pickle_to_location: df',data=loc)
        data.to_pickle(loc)

    else:
        pf = open(loc, "wb")
        dbg('storage.write_pickle_to_location: general data',data=loc)
        pf.dump(data, pickle_out)
        pf.close()
    return data



def write_df_to_location(df, loc ):
    '''
    loc can be s3 location like s3://bucket/filename.csv
    or a regular filepath /usr/local/thing.bin
    '''
    if is_s3_location(loc):
        bucket, key =parse_s3_location(loc)
        dbg('storage.write_df_to_location: s3',data=[bucket,key])
        write_df_to_s3(df,bucket,key)
    else:
        dbg('storage.write_df_to_location: file',data=loc)
        df.to_csv(loc, index=False)
    return

def read_df_from_location(loc):
    '''
    loc can be s3 location like s3://bucket/filename.csv
    or a regular filepath /usr/local/thing.bin
    '''
    if is_s3_location(loc):
        bucket, key = parse_s3_location(loc)
        dbg('storage.read_df_from_location: s3',data=[bucket,key])
        df = read_df_from_s3(bucket,key)
    else:
        #print("file")
        dbg('storage.read_df_from_location: file',data=loc)
        df = pd.read_csv(loc)
    return df




def load_agent_location(agent_class, path):
    if is_s3_location(path):
        # copy the file locally
        bucket, key = parse_s3_location(path)
        dbg('storage.load_agent_location: s3',data=[bucket,key])
        s3_client = boto3.client('s3')
        dest = slugify(key)
        dbg('storage.load_agent_location: download to', data=dest)
        s3_client.download_file( bucket, key, dest)
        path = dest

    dbg('storage.load_agent_location: file', data=path)
    agent = agent_class.load(path)
    return agent


def save_agent_location(agent,path):
    org_path = path
    if is_s3_location(path):
        # copy the file locally
        bucket, key = parse_s3_location(path)
        path=key
    agent.save(path)
    if is_s3_location(org_path):
        s3_client = boto3.client('s3')
        local_file = key
        dbg('storage.save_agent_location: s3', data=(path,key) )
        s3_client.upload_file( local_file, bucket, key)




def test_write_df_to_s3_location():
    df=pd.DataFrame( {'a':[1,2,999],'b':['yes','no','always']})
    write_df_to_location(df, "s3://example-telepath-bucket/df.csv")
    write_df_to_location(df, "df.csv",)
    del df
    df=read_df_from_location("s3://example-telepath-bucket/df.csv")
    #print(df)

#test_write_df_to_location()
#exit(0)
