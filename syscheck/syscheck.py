import subprocess
import sys
import torch as th
import boto3
import fire

def proc_exec( bash_list, capture_output=True ):
    if capture_output:
        proc = subprocess.run( bash_list, capture_output=capture_output )
    else:
        proc = subprocess.run( bash_list, capture_output=capture_output, stderr=sys.stderr, stdout=sys.stdout)
    return(proc.stdout.decode('ascii'))

def dbg(s,data=None):
    if data:
        print('[dbg] '+s,data)
    else:
        print('[dbg] '+s)


def write_data_to_s3(data, bucket, key):
    s3_res = boto3.resource("s3")
    dbg('storage.write_data_to_s3', data=[bucket, key])
    s3_res.Object(bucket, key).put(Body=data)

def read_data_from_s3( bucket, key):
    s3_res = boto3.resource("s3")
    dbg('storage.read_data_from_s3', data=[bucket, key] )
    res = s3_res.Object(bucket, key).get()['Body'].read()
    res = res.decode('ascii')
    print(res)
    return res


class SyscheckCmd:
    def s3check(self,bucket,key):
        print("s3check")
        data = 'xyzabc'
        write_data_to_s3( data, bucket, key)
        dat = read_data_from_s3( bucket, key)
        dbg("check if read and write match")
        if dat==data:
            print("looks good")
        else:
            print(dat+" does not match "+data)

    def deepcheck(self):
        print('torch.__version__', th.__version__)
        print('torch.version.cuda', th.version.cuda)
        print(proc_exec(['uname', '-a']))
        print(proc_exec(['cat', '/etc/issue']))
        print(proc_exec(['free', '-h']))
        print(proc_exec(['lscpu']))
        print(proc_exec(['nvidia-smi']))


def header():
    print("telepath syscheck")
    print("(c) Pathway Intelligence Inc")
    print("---")

if __name__=="__main__":
    header()
    fire.Fire(SyscheckCmd)
