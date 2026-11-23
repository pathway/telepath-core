import vaex as vx
import dask.dataframe as dd
import time
import pandas as pd
from block_timer.timer import Timer
import random

print("loading frame...")

# rows: 3,448,439
fn='../worker/test/test_data/bigdrive_test_data/generated_long_8.csv'
fnh='../worker/test/test_data/bigdrive_test_data/generated_long_8.csv.hdf5'
fna='../worker/test/test_data/bigdrive_test_data/generated_long_8.csv.arrow'

fnp='/path/to/telepath-runs/cartpole/replay/extended_replay4.10000000.csv.parquet' # errors out

def do_convert(csv):
    dd.read_csv(csv)
    # df = vaex.from_csv('./my_data/my_big_file.csv', convert=True,

with Timer(title="convert"):
    do_convert(fn)

with Timer(title="open"):
    tfn = fn
    print(tfn)
    #vdf = vx.open(tfn)
    vdf = dd.read_csv(tfn,)
    print(vdf.shape)
    print(vdf.dtypes)

with Timer(title="describe"):
    print( vdf.describe() )

if 0:
    with Timer(title="describe.compute"):
        print( vdf.describe().compute() )

#pd.read_csv('some_file.csv')


#vdf.export_arrow(fna)
#vdf.export_hdf5(fnh)
#vdf.export_parquet(fnp)

def getsamples(vdf, start=0, range_size=10000, batch_size=32,reps=100 ):
    start = start
    print( "start,range_size",start,range_size)
    for i in range(0, reps):
        slice = vdf.head(start+range_size).tail(range_size)
        # print(slice.shape)
        with Timer(title="slice.sample"):
            res = vdf.sample(frac=0.0001) #batch_size/vdf.shape[0].compute())
            print(res.shape[0].compute())
        #print(res.describe())
        # print(res)


def get_sample_time(reps=100,range_size=10**4, batch_size=256):
    dic={'reps':reps,
         'range_size':range_size,
         }
    print('range_size',range_size)
    starts=[0, 10000, ] #100000, 1000000, 10000000]
    random.shuffle(starts)
    print(starts)
    for a in range(0,1):
        all_times=[]
        for start in starts:
            #t_s = time.time()
            with Timer(title="start_from_" + str(start)):
                #range_size = 2000000
                print(start)
                t_s = time.time()
                with Timer(title="getsamples"):
                    getsamples(vdf, start=start, range_size=range_size,batch_size=batch_size, reps=reps)
                t_e = time.time()
                t_delta = (t_e-t_s)*1000
                all_times.append(t_delta)

        dic['s_100_minibatch'] = sum(all_times)/len(all_times)
        dic['batch_size'] = batch_size
        return dic

reps = 10
batch_size = 256

df = pd.DataFrame()
dics = []

for range_size in [  10**2, 10**4,]: # 10**5, 10**6]:
    for batch_size in [16]: #,64,256,2048]:
        dic = get_sample_time( reps, range_size, batch_size )
        dics.append(dic.copy())
            #t_e = time.time()
        #print("sample 50*10 latency", (t_e - t_s) * 1000.0, )


df=pd.DataFrame(dics)
print(df)

'''
    reps  range_size  s_100_minibatch  batch_size
0    100       10000       236.348200          16
1    100       10000       243.797636          64
2    100       10000       245.102978         256
3    100       10000       241.033220        2048
4    100      100000       359.073496          16
5    100      100000       380.879402          64
6    100      100000       345.986080         256
7    100      100000       358.190536        2048
8    100     1000000      2224.238729          16
9    100     1000000      2204.707146          64
10   100     1000000      2406.683493         256
11   100     1000000      2294.285536        2048

'''



'''
Conclusions:
- csv is hopelessly slow for open
- hdf5 + arrow is super fast for open
- parquet often failing no idea why : SEGSEGV

- sampling can be handled using head+tail to focus sampling on certain chunks
- sampling within head+tail chunks fast in every format including csvs. Same speed on all formats.

- csv: we dont even know row count in advance, to nicely handle windows
- range_size makes a huge difference
    - 10**4 : 0.18s per 100 samples
    - 10**5 : 0.33s per 100 samples
    - 10**6 : 1.9s per 100 samples
    - 10**7 : 14 s per 100 samples
'''
