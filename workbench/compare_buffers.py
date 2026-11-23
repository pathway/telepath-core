import vaex as vx
import time
import pandas as pd
from block_timer.timer import Timer
import random

print("loading vx...")

# rows: 3,448,439
fn='/path/to/telepath-runs/cartpole/replay/extended_replay4.10000000.csv'
fna='/path/to/telepath-runs/cartpole/replay/extended_replay4.10000000.csv.arrow'
fnh='/path/to/telepath-runs/cartpole/replay/extended_replay4.10000000.csv.hdf5'

fnp='/path/to/telepath-runs/cartpole/replay/extended_replay4.10000000.csv.parquet' # errors out


with Timer(title="open"):
    tfn = fna
    print(tfn)
    vdf = vx.open(tfn)
    print(vdf.dtypes)
    print(vdf.shape)

with Timer(title="describe"):
    print( vdf.describe() )
#pd.read_csv('some_file.csv')


#vdf.export_arrow(fna)
#vdf.export_hdf5(fnh)
#vdf.export_parquet(fnp)

def getsamples(vdf, start=0, range_size=10000, batch_size=32, ):
    start = start
    for i in range(0, 100):
        slice = vdf.head(start+range_size).tail(range_size)
        # print(slice.shape)
        res = slice.sample(n=batch_size)
        #print(res.describe())
        # print(res)


range_size = 10**6
print('range_size',range_size)

starts=[0, 10000, 100000, 1000000, 10000000]
random.shuffle(starts)
print(starts)
for a in range(0,10):
    for start in starts:
        #t_s = time.time()
        with Timer(title="start_from_" + str(start)):
            #range_size = 2000000
            getsamples(vdf, start=start, range_size=range_size,batch_size=32)
        #t_e = time.time()
        #print("sample 50*10 latency", (t_e - t_s) * 1000.0, )


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
