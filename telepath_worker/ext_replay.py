import pandas as pd
import numpy as np
import torch as th
import time
import random
import matplotlib.pyplot as plt
from pprint import pprint as pp

from stable_baselines3.common.type_aliases import ReplayBufferSamples

from telepath_worker import telepath_utils
from telepath_worker import spec
from telepath_gym import simplegym
from telepath_worker import vx_shift

class ReplayReturn():
    def __str__(self):
        s=''
        if hasattr(self, 'observations'):
            s+= "\nobs\n"+str(self.observations) +"\n"

        if hasattr(self, 'next_observations'):
            s+= "\nnext_obs\n"+str(self.next_observations)

        if hasattr(self, 'actions'):
            s+= "\naction\n"+str(self.actions)

        if hasattr(self, 'dones'):
            s+= "\ndone\n"+str(self.dones)

        if hasattr(self, 'rewards'):
            s+= "\nreward\n"+str(self.rewards)

        return s

def _get_default_fieldmap():
    fieldmap = {}
    fieldmap['obs_cols'] = []
    fieldmap['next_obs_prefix'] = 'nx_'
    fieldmap['next_obs_cols'] = []
    fieldmap['post_event_cols'] = []
    fieldmap['agent'] = None
    fieldmap['sid'] = None
    fieldmap['tid'] = None
    fieldmap['step'] = None
    fieldmap['reward'] = 'reward'
    fieldmap['done'] = 'done'
    fieldmap['action'] = 'action'
    return fieldmap


class GenDataFrameBuffer():

    df = None
    df_reader = None
    replay_loc = None
    lazy_dict_list = None
    subs = None

    # settings
    chunksize = 100000

    # fieldmap
    fieldmap = None

    # columns

    def __init__(self, replay_loc):
        # doesnt neccessarily load it, but needs a path, or s3 location
        self.replay_loc = replay_loc

        self.fieldmap = _get_default_fieldmap()

    def get_col_subsets(self):
        subs = {}

        single_cols = [ 'reward',
                        'done',
                        'action',
                        'agent' ,
                        'tid' ,
                        'step' ,
                        'sid',
                        ]

        for config_fld_name in single_cols:
            subs[config_fld_name] = 0
            if config_fld_name not in self.fieldmap:
                subs[ config_fld_name ] = 0
            elif self.fieldmap[config_fld_name] in list(self.df):
                subs[ config_fld_name ] = 1
            else:
                subs[ config_fld_name] = -1

        multi_cols = [ 'obs_cols',
                       'next_obs_cols',
                       'post_event_cols',
                       ]

        for config_fld_name in multi_cols:
            subs[config_fld_name] = 0
            for c in self.fieldmap[config_fld_name]:
                if c in list(self.df):
                    subs[config_fld_name]+=1
                else:
                    subs[config_fld_name]=-1
                    break

        self.subs = subs
        return subs

    def get_col_validity(self):
        valid=[]
        subs = self.get_col_subsets()
        # direct train
        valid={
            'trace':0,
            'trace_sid': 0,
            'trace_traj': 0,
            'agent_policy_trace': 0,
            'policy_trace': 0,
            'train_model': 0,
            'train_rl': 0,
            'train_bandit': 0,
        }

        if subs['tid']==1 and subs['step']==1:
            valid['trace_traj']=1
        if subs['sid']==1:
            valid['trace_sid']=1
        if valid['trace_sid'] or valid['trace_traj']:
            valid['trace']=1

        if subs['action']==1 and valid['trace']:
            valid['policy_trace']=1
        if subs['agent']==1:
            valid['agent_policy_trace']=1

        if subs['obs_cols']>0 and subs['reward']==1 and subs['action']==1:
            valid['train_bandit']=1
        if subs['obs_cols']>0 and subs['reward']==1 and subs['next_obs_cols']>0 and subs['action']==1:
            valid['train_rl']=1
        if subs['obs_cols']>0 and subs['reward']==1 and subs['next_obs_cols']>0 and subs['action']==1 and subs['done']==1:
            valid['train_model']=1

        return valid

    def _set_obs_cols__default(self,obs_shape_0):
        if self.fieldmap['obs_cols'] is None:
            self.fieldmap['obs_cols'] = []
        if not self.fieldmap['obs_cols']:
            for oi in range(0, obs_shape_0):
                self.fieldmap['obs_cols'].append('obs'+str(oi))

    def _set_nx_obs_cols__default(self,obs_shape_0):
        if self.fieldmap['next_obs_cols'] is None:
            self.fieldmap['next_obs_cols'] = []
        if not self.fieldmap['next_obs_cols']:
            for oi in range(0, obs_shape_0):
                self.fieldmap['next_obs_cols'].append( self.fieldmap['next_obs_prefix'] + 'obs'+str(oi))



    def _guess_obs_cols(self):
        cols = list(self.df)
        self.fieldmap['obs_cols'] = [ cc for cc in cols if cc.startswith('obs') ]
        self.fieldmap['next_obs_cols'] = [ cc for cc in cols if cc.startswith('nx_obs') ]

    def start(self):
        pass

'''
    def add(self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray) -> None:
            ):
'''



def get_specgym_from_replay_spec( replay_spec_object ):
    na = replay_spec_object.spec['n_actions']
    nobs = len(replay_spec_object.spec['fieldmap']['obs_cols'])
    gym = simplegym.SimpleGym_DiscreteAction_BoxObs()
    gym.init(na, nobs)
    return gym



def list_to_tensor( lst ):
    return th.tensor(np.array([[s] for s in lst]), dtype=th.float)



import vaex as vx


class IncrementalReader:

    df_reader = None
    batch_size = 32

    def _reset_reader(self):
        if self.source == 'mem':
            return
        self.df_reader = pd.read_csv(self.replay_loc, chunksize=self.chunksize)

    def touch_reader(self):
        if self.source == 'mem':
            return

        if self.df_reader is None:
            self._reset_reader()

        self.max_samples_from_chunk = self.chunksize // self.batch_size

        if self.df is None or (self.df.shape[0] - self.start_index <= self.batch_size ) or (self.samples_from_chunk>0 and self.samples_from_chunk >= self.max_samples_from_chunk):
            print("[buffer] chunkread", self.samples_from_chunk)
            if self.df is not None:
                print( "start_index {}\tbatch_size {}\tdf.shape[0] {}".format( self.start_index, self.batch_size,  self.df.shape[0]))

            tries = 0
            while tries < 2:
                try:
                    self.next_chunk()
                    break
                except StopIteration:
                    if tries == 0:
                        self._reset_reader()
                        tries += 1
                        break
                    else:
                        pass

            self.start_index = 0
            self.samples_from_chunk = 0

    def next_chunk(self):
        if self.source == 'mem':
            return
        self.df = self.df_reader.get_chunk(self.chunksize)

    def needs_new_chunk(self, batch_size):
        if self.source == 'mem':
            return False

        if self.df is None or \
            (self.df.shape[0] - self.start_index <= batch_size ) or \
            (self.samples_from_chunk>0 and self.samples_from_chunk >= self.max_samples_from_chunk):
            return True

        return False

    def chunk_writer(self, out_path):
        df_chunks = pd.read_csv(self.replay_loc, sep=',', skiprows=(0,), header=(1), chunksize=self.chunksize)
        header = True

        # add shifted nx_obs
        # we could do this with sequential sampler

        for chunk in df_chunks:
            chunk.to_csv(out_path,header=header, mode='a')
            header = False



class LazyReader:
    # LazyWriter?

    def reconstruct(self):
        if not self.df:
            self.df = pd.DataFrame.from_dict(self.lazy_dict_list)
        else:
            self.df = self.df.append( pd.DataFrame.from_dict(self.lazy_dict_list) )

    def lazy_add(self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            sids: list=None,
            tids: list=None,
            steps: list=None,
            ) -> None:
        # Store in memory lists for now, for speed

        row = {
        'action': action[0],
        'reward' : reward[0],
        'done': done[0]
        }
        if sids:
            row['sid'] = sids[0]
        else:
            row['sid'] = telepath_utils.get_rand_sampleid()

        if tids:
            row['tid'] = tids[0]
            row['step'] = steps[0]

        # TODO: this isnt quite right
        if not self.fieldmap['obs_cols']: self._set_obs_cols__default(obs.shape[0])
        if not self.fieldmap['next_obs_cols']: self._set_nx_obs_cols__default(obs.shape[0])

        #print( "self.fieldmap['obs_cols']", self.fieldmap['obs_cols'] )
        #print("obs.shape[0]", obs.shape[0])

        for oi in range(0,obs.shape[0]):
            row[ self.fieldmap['obs_cols'][oi] ] = obs[oi]

        for oi in range(0,obs.shape[0]):
            row[ self.fieldmap['next_obs_cols'][oi] ] = next_obs[oi]

        #'obs': np.array(obs).copy(), 'next_obs': np.array(next_obs).copy(),
        #self.df = self.df.append( row, ignore_index=True )
        if self.lazy_dict_list is None:
            self.lazy_dict_list = []
        self.lazy_dict_list.append( row )

        #print(self.df)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0    # wrap around; this doesnt actually work



class SpecReplay:

    specs = None

    def set_spec(self, spec_loc ):
        self.specs = spec.ReplaySpec()
        self.specs.load(spec_loc)
        #print(self.specs)

        sp = self.specs.spec

        self.fieldmap = self.specs.spec['fieldmap']
        '''
        if 'sample_strategy' in sp: self.sample_strategy=sp['sample_strategy']
        if 'chunksize' in sp: self.chunksize=sp['chunksize']

        if 'agent' in sp: self.fieldmap['agent']=sp['agent']
        if 'reward' in sp: self.fieldmap['reward']=sp['reward']
        if 'done' in sp: self.fieldmap['done']=sp['done']
        if 'action' in sp: self.fieldmap['action']=sp['action']
        if 'sid' in sp: self.fieldmap['sid'] = sp['sid']
        if 'tid' in sp: self.fieldmap['tid']=sp['tid']
        if 'step' in sp: self.fieldmap['step']=sp['step']

        if 'post_event_cols' in sp: self.fieldmap['post_event_cols']=sp['post_event_cols']

        if 'obs_cols' in sp: self.fieldmap['obs_cols']=sp['obs_cols']
        if 'next_obs_cols' in sp: self.fieldmap['next_obs_cols']=sp['next_obs_cols']
        '''

        # set next_obs_cols
        if 'next_obs_cols' not in self.fieldmap or len(self.fieldmap['next_obs_cols'])==0:
            self.fieldmap['next_obs_cols']=[]
            for obs_col in self.fieldmap['obs_cols']:
                self.fieldmap['next_obs_cols'].append( self.fieldmap['next_obs_prefix'] + obs_col )

    def get_specgym(self):
        return get_specgym_from_replay_spec( self.specs )



class PdPlainBuffer(GenDataFrameBuffer,LazyReader,IncrementalReader,SpecReplay):
    # Stores obs in individual columns
    pos = 0
    buffer_size = 100
    full = False
    start_index = 0
    samples_from_chunk = 0
    source = None   # 'mem', 'file'; if mem then ignore chunking behaviour
    sample_strategy = None  # 'rand','seq'

    def __init__(self,replay_loc,sample_strategy='rand',source='file'):
        super().__init__(replay_loc=replay_loc,)
        self.sample_strategy=sample_strategy
        self.source=source

    def compute_virtual_cols(self):
        # not implemented
        pass

    def start(self):
        pass

    def print_config(self):
        print("samples_from_chunk",self.samples_from_chunk)
        print("start_index",self.start_index)
        print("sample_strategy",self.sample_strategy)
        print("df",self.df)

    def get_mem_usage(self):
        mem = self.df.memory_usage(index=True, deep=True)
        return { 'total_mb':mem.sum()/(1024*1024), 'cols_mb':mem/(1024*1024)  }

    def sample(self, batch_size, env=None):
        """
        :param batch_size: (int) Number of element to sample
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: (Union[RolloutBufferSamples, ReplayBufferSamples])
        """

        self.batch_size=batch_size
        self.touch_reader()

        #print("+PdPlainBuffer: sample")
        if not self.fieldmap['obs_cols']:
            self._guess_obs_cols()

        # print( 'replay columns', list(self.df))

        sample = None

        if self.sample_strategy=="seq":
            start = self.start_index
            end = self.start_index+batch_size
            # went past the end?
            if end > self.df.shape[0]:
                end=self.df.shape[0]

            sample = self.df.iloc[start:end]

            # loop over?
            self.start_index += batch_size
            if self.start_index + batch_size > self.df.shape[0]:
                self.start_index=0


        elif self.sample_strategy=="rand":
            sample = self.df.sample(batch_size)

        #print('sample',sample)
        #print('self.fieldmap['reward']',self.fieldmap['reward'])
        self.samples_from_chunk +=1

        ret = ReplayReturn()
        #import pdb; pdb.set_trace()
        ret.rewards = list_to_tensor( sample[self.fieldmap['reward']].tolist() ).cuda()
        ret.dones = list_to_tensor( sample[self.fieldmap['done']].tolist()  ).cuda()
        ret.actions = list_to_tensor(  sample[self.fieldmap['action']].tolist() ).to(th.int64).cuda()

        obslist= [  th.tensor(sample[oc].tolist(),dtype=th.float)  for oc in self.fieldmap['obs_cols'] ]
        ret.observations = th.stack( obslist, dim=1,).cuda()

        next_obslist= [  th.tensor(sample[oc].tolist(),dtype=th.float)  for oc in self.fieldmap['next_obs_cols'] ]
        ret.next_observations = th.stack( next_obslist, dim=1).cuda()

        #print("-PdPlainBuffer: sample")
        return ret


    def export_format(self, fn):
        #self.df.to_parquet(fn)
        #self.df.to_hdf(fn,key='main')
        self.df.to_feather(fn,)


    def _buffer_perf_stats(self, batch_size = 32):
        '''
        in: buffer_id
        out: buffer_describe df
        '''
        # create replay buffer
        #common_replay = self.READ_BUFFER_TYPE()  # why two?
        #dbg("buffer_stats", comment="load replay", data=replay_loc )

        print("loading vx...")

        #if self.replay_loc.endswith('.csv'):
        #    vdf=vx.from_csv(self.replay_loc)
        #elif
        # self.replay_loc.endswith('.parquet'):
        t_s = time.time()
        vdf = vx.open(self.replay_loc)

        print(vdf.shape)
        print(vdf.column_names)
        t_e = time.time()
        print( "read latency ms", (t_e-t_s)*1000.0,)
        #print(vdf.describe() )
        '''
        Formats:
        parquet chokes
        When sample500 is from random across the file:

        csv 26s   6s sample500
        arrow: 4 ms     580 ms sample500
        hdf5:  7ms       6s sample500 

        When sample500 is done from big chunks:

        '''
        def getsamples(vdf, start=0, range_size=10000, sample_reps=10, batch_size=32,):
            start = start
            for i in range(0, sample_reps):
                slice = vdf.head(start).tail(range_size)
                #print(slice.shape)
                res=slice.sample(n=batch_size)
                #print( res.describe() )
                #print(res)

        sample_reps = 10
        for start in [0,10**4,10**5,10**6,10**7,10**8,]:
            t_s = time.time()
            getsamples( vdf, start=start, range_size=10**5,batch_size=batch_size, sample_reps=sample_reps)
            t_e = time.time()
            print( "sample per batch*{} samples latency {} ".format(batch_size, sample_reps,(t_e-t_s)*1000.0, ), )

        return
        print("exporting vx...")
        vdf.export_parquet(self.replay_loc+'.parquet')
        vdf.export_arrow(self.replay_loc+'.arrow')
        #vdf.export_fits(self.replay_loc+'.fits')
        # see https://vaex.readthedocs.io/en/latest/faq.html#I-have-a-massive-CSV-file-which-I-can-not-fit-all-into-memory-at-one-time.-How-do-I-convert-it-to-HDF5?
        vdf.export_hdf5(self.replay_loc+'.hdf5')
        print("vdf describe")
        print(vdf.shape)

        print("CHUNK STATS ONLY -- switch to vaex")
        self.start()
        self.sample(batch_size=1)
        #self.df = storage.read_df_from_location(self.sett['replay_loc'])
        self._guess_obs_cols()

        #print( self.df.describe() )
        print("shape")
        print(self.df.shape)

        print("columns")
        print(self.df.columns)

        print("obs_cols")
        print(self.fieldmap['obs_cols'])
        print( self.df[ self.fieldmap['obs_cols'] ].describe() )

        print("next_obs_cols")
        print(self.fieldmap['next_obs_cols'])
        print( self.df[ self.fieldmap['next_obs_cols'] ].describe() )

        print("action")
        d1= pd.DataFrame( self.df.action.value_counts().head(10) )
        d2= pd.DataFrame( self.df.action.value_counts(normalize=True).head(10) )
        print( d1.merge( d2,left_index=True, right_index=True, suffixes=( '_count', '_ratio')  ).sort_index() )

        print("done")
        d1=pd.DataFrame( self.df.done.value_counts().head(10) )
        d2=pd.DataFrame( self.df.done.value_counts(normalize=True).head(10) )
        print( d1.merge( d2,left_index=True, right_index=True, suffixes=( '_count', '_ratio')  ) )

        #print( pd.DataFrame( self.df.done.value_counts().head(10) ).merge( self.df.done.value_counts(normalize=True).head(10) ) )
        print("reward")
        print( self.df.reward.describe() )
        print( self.df.reward.value_counts().head(10))

        #fn_buffer_describe='replay_describe.csv'
        #dbg("buffer_stats", comment="writting buffer describe", data=fn_buffer_describe)
        #print( self.df.describe().to_csv(fn_buffer_describe) )




def csv_stats(csv_fn):
    print(csv_fn)
    pass


from telepath_worker.chunk_sampler import ChunkSampler

class VxBuffer( GenDataFrameBuffer, SpecReplay ):

    pos = 0
    buffer_size = 10000
    full = False
    start = 0
    end = None
    sample_strategy = None  # 'rand','seq'
    device = 'cuda:0'

    chunk_sampler = None

    def read_from_location(self,):
        self.df = vx.from_csv( self.replay_loc, )# copy_index=True, index_col=0 )      # should use chunking?

        self.chunk_sampler = ChunkSampler( max_index = self.df.shape[0], chunk_size=100000, batch_size=32, chunk_resample_rate=0.04 )

        self.compute_virtual_cols()
        return True


    def _guess_obs_cols(self):
        cols = list(self.df)
        self.fieldmap['obs_cols'] = [ cc for cc in cols if cc.startswith('obs') ]
        self.fieldmap['next_obs_cols'] = [ cc for cc in cols if cc.startswith('nx_obs') ]

    def shuffle(self):
        self.df = self.df.sample(frac=1, replace=False )

    def traj_order(self):
        if 'tid' in self.fieldmap and 'step' in self.fieldmap:
            self.df = self.df.sort( by=[self.fieldmap['tid'],self.fieldmap['step'] ]  )

    def minmax(self):
        df_new = self.df.copy()
        cl = self.df.columns
        for c in cl:
            df_new[c ] = ( self.df[c] - (self.df[c].min()) ) / (  self.df[c].max() - self.df[c].min()  )
            print(c)
        self.df = df_new

    def normalize(self):
        df_new = self.df.copy()
        cl = self.df.columns
        for c in cl:
            df_new[c ] = ( self.df[c] - (self.df[c].mean()) ) / (  self.df[c].max() - self.df[c].min()  )
            print(c)
        self.df = df_new

    def _sample_df_to_return(self,sample):
        ret = ReplayReturn()
        ret.rewards = th.tensor( np.array( [ [s] for s in sample[self.fieldmap['reward']].tolist()] ),dtype=th.float, device=self.device)

        if 'done' in self.fieldmap and self.fieldmap['done']:
            ret.dones = th.tensor( np.array( [ [s] for s in sample[self.fieldmap['done']].tolist()] ),dtype=th.float, device=self.device )
        else:
            ret.dones = th.tensor( np.zeros( sample.shape[0] ) )

        ret.actions = th.tensor( np.array( [ [s] for s in sample[self.fieldmap['action']].tolist()] ),dtype=th.int64, device=self.device)

        obslist= [  th.tensor(sample[oc].tolist(),dtype=th.float, device=self.device)  for oc in self.fieldmap['obs_cols'] ]
        ret.observations = th.stack( obslist, dim=1, )

        next_obslist= [  th.tensor(sample[oc].tolist(),dtype=th.float, device=self.device)  for oc in self.fieldmap['next_obs_cols'] ]
        ret.next_observations = th.stack( next_obslist, dim=1,)
        return ret

    def sample_seq(self, batch_size, env=None):
        sample = self.df.head(self.start+batch_size).tail(batch_size)
        #self.compute_virtual_cols()
        self.start+=batch_size
        if self.start + batch_size > self.df.shape[0]:
            self.start=0
        ret = self._sample_df_to_return(sample)
        return ret

    def sample_random(self, batch_size, env=None):
        # TODO: Simplify to speed this up
        """
        :param batch_size: (int) Number of element to sample
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return: (Union[RolloutBufferSamples, ReplayBufferSamples])
        """
        if not self.fieldmap['obs_cols']:
            self._guess_obs_cols()

        sp = self.chunk_sampler.next_sample_params()

        slice = self.df.head(sp['chunk_start_index'] + self.chunk_sampler.chunk_size).tail(self.chunk_sampler.chunk_size)

        sample = slice.sample(batch_size) # inefficient. see
        ret = self._sample_df_to_return(sample)
        return ret

    def sample(self,batch_size, env=None):
        #print("Vx sample")
        if self.sample_strategy=='seq':
            return self.sample_seq( batch_size, env)
        else:
            return self.sample_random( batch_size, env)


    def print_colstats(self):
        for c in list(self.df.columns):
            print(c)
            print(self.df[c].value_counts() )
            print('--\n\n')

        for c in list(self.df.virtual_columns):
            print(c)
            print(self.df[c].value_counts() )
            print('--\n\n')


    def compute_virtual_cols(self):
        # must happen after read
        if self.specs is None:
            return
        sp = self.specs.spec

        #return

        if not self.fieldmap['obs_cols']: self._set_obs_cols__default( len(sp['obs_cols']) )
        if not self.fieldmap['next_obs_cols']: self._set_nx_obs_cols__default( len(sp['obs_cols']) )


        # first do column transformations in order
        if 'expr_cols' in sp['fieldmap']:
            for key_field in sp['fieldmap']['expr_cols'].keys():
                val = sp['fieldmap']['expr_cols'][key_field]
                if val=='boolint':
                    self.df[key_field] = self.df[key_field].astype('int')

                elif val=='dowtrig':
                    newfn=key_field+'_sin'
                    self.df[newfn] = (2.0*np.pi*self.df[key_field]/7.0 ).sin()
                    newfn=key_field+'_cos'
                    self.df[newfn] = (2.0*np.pi*self.df[key_field]/7.0 ).cos()

                #if val=='log10':
                else:
                    #self.df[key_field] = np.log10( self.df[key_field] )
                    #self.df[key_field] = np.log10( self.df[key_field] )
                    self.df.add_virtual_column(key_field, val)


        # then consider the reward expression
        if 'reward_expr' in sp['fieldmap']:
            reward_col_name="reward_expr"
            self.df.add_virtual_column(reward_col_name, self.specs.spec['fieldmap']['reward_expr'] )
            self.fieldmap['reward'] = reward_col_name

        #print(list(self.df.columns))
        #print(list(self.df.virtual_columns))
        # see https://stackoverflow.com/a/61075267
        #self.df['step_shift']=self.df.cust1.shift_constant('step',1)
        #print(self.df['virt_reward_shift'])


    def plots_2d(self, dir, fields=None, forbid_fields=[] ):
        # mapp to avoid repeats
        mapp = {}
        if fields is None:
            fl = list(self.df)
        else:
            fl = fields
        for c in fl:
            for d in fl:
                if c==d:
                    continue
                if c in forbid_fields or d in forbid_fields:
                    continue
                print(c,d)
                if (c,d) in mapp.keys():
                    #print("found")
                    continue
                fn = 'plot2d_' + c +'-'+d+'.png'
                self.df.plot(c, d, hardcopy=dir+fn,f='log10', shape=8)
                #plt.show()

                mapp[(c,d)]=fn
                mapp[(d,c)]=fn

    def plots_1d(self, dir, fields=None, forbid_fields=[]):
        if fields is None:
            fl = list(self.df)
        else:
            fl = fields
        for c in fl:
            if c in forbid_fields:
                continue
            print(c)
            fn = 'plot1d_'+c+'.png'
            self.df.plot1d(c ,hardcopy=dir+fn, shape=64)
            self.df.plot1d(c ,hardcopy=dir+fn, alpha=0.5,shape=8)
            #plt.show()

'''
ISSUE: We need a calculus of columns.

- shifted obs/evt columns (+/- 1,2,3..)
- computed obs columns (?)
- shifted computed obs/evt columns (?)
- nx_obs ...
- computed reward

Questions:
- what order do these things happen?
- handling dependencies between them
- some of these require the rows to be in fully sorted order, but in the end we need random order

Plans
---------
V0
- augment phase
    - sort the whole file
    - add shifted columns
    - add preliminary computed columns
    - fully randomize order
    - do we cache at this point?? with a faster format??
- sample phase
    - sample; add final computed columns per sample


'''



'''
ISSUE: VxBuffer vs PdPlainBuffer
'''

'''
ISSUE: Normalizing buffers

See:
# Automatically normalize the input features and reward
env = VecNormalize(env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)


'''
