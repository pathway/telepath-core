from telepath_worker.telepath_utils import *

def test_idgen():
    id = get_rand_agentid(); print(id)
    id = get_rand_sampleid(); print(id)
    id = get_rand_trajid(); print(id)
    id = get_rand_id(prefix='z', length=16); print(id)

def test_dbg():
    dbg('section','comment', data={'data':1,'item':'apple'})

if __name__=="__main__":
    test_idgen()
    test_dbg()
