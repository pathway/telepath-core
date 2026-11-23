from telepath_worker import ext_replay
from pprint import pprint as pp
import pandas as pd
import pdb
import numpy as np
import torch as th

replay_csv_file_small='test/test_data/generated_tiny_cartpole.csv'
replay_csv_file_huge='test/test_data/generated.csv'  # see generate_replay.py


# ------ vx
def _load_small_replay_vx():
    r = ext_replay.VxBuffer(replay_csv_file_small)
    r.sample_strategy='seq'
    r.fieldmap['action']='action'
    r.fieldmap['reward']='rew'
    r.fieldmap['obs_cols']=['obs0','obs1','obs2','obs3']
    r.fieldmap['next_obs_cols']=['nx_obs0','nx_obs1','nx_obs2','nx_obs3']
    r.read_from_location()
    return r

def __test_vxbuffer_normalize():
    r=_load_small_replay_vx()
    print(r.df)
    r.normalize()
    print(r.df)

def test_vxbuffer_shuffle():
    r=_load_small_replay_vx()
    r.shuffle()
    print(r.df)

def test_vxbuffer_basic():
    r=_load_small_replay_vx()
    print(r.df)
    for i in range(0,7):
        res = r.sample(2)
        print(res)



def test_vxbuffer_huge(): # slow
    r = ext_replay.VxBuffer(replay_csv_file_huge)
    r.read_from_location()

    r.fieldmap['reward']='rew'
    r.fieldmap['done']='done'
    r.fieldmap['action']='action'
    r.fieldmap['obs_cols']=['obs1']
    r.fieldmap['next_obs_cols']=['nx_obs1']

    batch_size=2
    res = r.sample(batch_size)
    assert res.observations.shape[0]==batch_size
    #print(res)




# -------- pdplain



def test_pdplainbuffer_basic():
    r = ext_replay.PdPlainBuffer(replay_csv_file_small)
    r.fieldmap['reward']='rew'
    #r.read_from_location()
    print(r)
    res = r.sample(2)
    pp(res)


def _plainbuffer_accept(csv_fn):
    r = ext_replay.PdPlainBuffer(csv_fn)    # doesnt use it, never reads it
    print(csv_fn)
    #r.reward='rew'
    '''
    r.done='done'
    r.action='action'
    r.obs_cols=['obs0','obs1','obs2','obs3',]
    r.next_obs_cols=['nx_obs0','nx_obs1','nx_obs2','nx_obs3',]
    '''

    #r.read_from_location()
    for i in range(0,5):
        obs = np.array([i,])
        next_obs = th.tensor([i+1,])
        action = [4,]
        reward = [5,]
        done = [1,]
        r.lazy_add( obs=obs, next_obs=next_obs, action=action ,reward= reward, done=done)
    r.reconstruct()
    r.source='mem'
    print(r)
    r.print_config()
    return r


def test_pdplainbuffer_seq():
    r = _plainbuffer_accept(replay_csv_file_small)
    r.sample_strategy = 'seq'
    for i in range(0,12):
        res = r.sample(2)
        print(res)
    print(r)


def test_pdplainbuffer_rand():
    r = _plainbuffer_accept(replay_csv_file_small)
    r.sample_strategy = 'rand'
    for i in range(0,12):
        res = r.sample(2)
        print(res)

    print(r)


def test_pdplainbuffer_accept():
    r = _plainbuffer_accept(replay_csv_file_small)
    print(r)
    print(r.df)
    res = r.sample(1)
    #assert res.observations[0]==th.tensor([[2]])
    #assert res.next_observations[0]==th.tensor([[3]])
    assert res.actions[0]==th.tensor([[4]])
    assert res.rewards[0]==th.tensor([[5]])
    assert res.dones[0]==th.tensor([[1]])
    print(res)



def test_pdplainbuffer_mem():
    r = ext_replay.PdPlainBuffer(replay_csv_file_small)    # doesnt use it, never reads it
    r.fieldmap['reward']='rew'
    r.sample(1)
    m=r.get_mem_usage()
    print(m)

    assert m['total_mb']>0.0
    assert m['total_mb']<0.1

    assert m['cols_mb']['action']>0.0
    assert m['cols_mb']['action']<0.1


def _pdplainbuffer_write():
    r = _plainbuffer_accept(replay_csv_file_small)
    print(r.df)
    r.export_format('/tmp/telepath_test__export.hdf5')


def test_pdplainbuffer_huge():
    rb = ext_replay.PdPlainBuffer(replay_csv_file_huge)

    rb.chunksize = 7
    rb.fieldmap['reward']='rew'
    rb.fieldmap['done']='done'
    rb.fieldmap['action']='action'
    rb.fieldmap['post_event_cols'] = ['obs1',]
    rb.fieldmap['obs_cols'] = ['obs1',]
    rb.fieldmap['next_obs_cols'] = ['nx_obs1',]

    #r.read_from_location()
    print(rb)
    res = rb.sample(2)
    #pdb.set_trace()
    subs = rb.get_col_subsets()
    print(subs)
    #pdb.set_trace()


def test_pdplainbuffer_huge_minichunk():
    '''
    Test efficient chunking
    '''
    r = ext_replay.PdPlainBuffer(replay_csv_file_huge)

    r.chunksize = 6
    r.fieldmap['reward'] = 'rew'
    r.fieldmap['done'] = 'done'
    r.fieldmap['action'] = 'action'

    # r.read_from_location()
    #print(r)
    batch_size=3
    offset=0
    res = r.sample(batch_size)
    assert res.observations.max().item() <= r.chunksize
    assert res.observations.min().item() >= 0

    offset+=batch_size
    res = r.sample(batch_size)
    assert res.observations.max().item() <= r.chunksize
    assert res.observations.min().item() >= 0

    offset+=batch_size
    res = r.sample(batch_size)
    assert res.observations.max().item() <= r.chunksize * 2
    assert res.observations.min().item() >= r.chunksize
    #pdb.set_trace()




# --------- deprecated? broken/

# slow
def __test_pandas_huge():
    r = pd.read_csv(replay_csv_file_huge)
    print(r)



if __name__=="__main__":
    print("MAIN")
    #test_pdplainbuffer_rand()
    test_pdplainbuffer_mem()
