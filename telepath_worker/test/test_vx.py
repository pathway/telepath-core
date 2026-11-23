import vaex as vx
from telepath_worker import ext_replay
from pprint import pprint as pp

replay_csv_file_small='test/test_data/generated_tiny_cartpole.csv'
replay_csv_file_huge='test/test_data/generated.csv'  # see generate_replay.py
replay_csv_file_client='test/test_data/client_example_01.csv'
replay_csv_file_client_spec='specs2/spec_replay_client_01.yaml'



def _test_vx_shift():
    r = ext_replay.VxBuffer(replay_csv_file_small)
    r.sample_strategy='seq'
    r.read_from_location()

    #print(r.df)
    #r.df['step_shift']=
    r.df.cust1.shift_constant(org_column='step',new_column='step_shift',n=-2,k=-1,inplace=True)

    print(r.df)


def _test_vx_shift_client():
    r = ext_replay.VxBuffer(replay_csv_file_client)
    r.sample_strategy='seq'
    r.set_spec(replay_csv_file_client_spec)

    pp(r.specs.spec)
    # can now safely read and sample; virtual cols added here
    r.read_from_location()
    r.traj_order()

    #r.df.rename('Unnamed: 0','Unnamed__0')
    print(r.df)

    r.df.cust1.shift_constant(org_column='account_id',new_column='account_id_unsh1',n=-1,k=-1,)
    r.df.cust1.shift_constant(org_column='days_since_kickoff',new_column='nx_days_since_kickoff',n=-1,k=-1,)
    #r.df.cust1.shift_constant(org_column='days_since_kickoff',new_column='days_since_kickoff_sh2',n=2,k=-1,)
    r.df['days_since_kickoff_delta1'] = r.df['nx_days_since_kickoff'] - r.df['days_since_kickoff']
    r.df['account_id_delta1'] = (r.df['account_id'] != r.df['account_id_unsh1']).astype('int')
    print( r.df)
    print( r.df[ r.df.account_id == r.df.account_id_unsh1 ]) #.days_since_kickoff_sh1 = -2
    replay = r
    #import pdb; pdb.set_trace()

    act = replay.df[ replay.df.account_id_delta1 == 0 ]
    act = act[ act.days_since_kickoff_delta1==1.0]
    print( act )
    #print( replay.df[ replay.df.account_id_delta1 == 0.0 ]  )

    print( r.df )

    # I want to make nx_* by just shifting all the obs columns.
    # But its not working out...
    for colname in r.fieldmap['obs_cols']:
        print(colname)
        # its not that simple with derived columns.  How to solve that?
        #if colname in r.df.virtual_columns:
        #    pass #r.df.cust1.shift_constant(org_column=colname, new_column=colname, n=-1, k=-1, )
        #    #r.df.rename()
        #else:
        #if colname in r.df._virtual_expressions:
        r.df.cust1.shift_constant(org_column=colname, new_column='nx_'+colname, n=-1, k=-1, )

    print(r.df)
    #r.df.cust1.shift_constant(org_column='account_id',new_column='account_id_sh1',n=1,k=-1,)

    # seq sampling fails here with:
    # r.sample_strategy='seq'
    # but rand sampling works
    '''
        Traceback (most recent call last):
      File "/path/to/telepath-core/worker/runtest.py", line 7, in <module>
        test.test_vx.test_vx_shift_client()
      File "/path/to/telepath-core/worker/test/test_vx.py", line 73, in test_vx_shift_client
        res = r.sample(2)
      File "/path/to/telepath-core/worker/ext_replay.py", line 733, in sample
        return self.sample_seq( batch_size, env)
      File "/path/to/telepath-core/worker/ext_replay.py", line 709, in sample_seq
        sample = self.df.head(self.start+batch_size).tail(batch_size)
      File "/home/user/miniconda3/envs/telepath3/lib/python3.8/site-packages/vaex/dataframe.py", line 3457, in tail
        return self[max(0, N - n):min(len(self), N)]
      File "/home/user/miniconda3/envs/telepath3/lib/python3.8/site-packages/vaex/dataframe.py", line 4626, in __getitem__
        df = self.trim()
      File "/home/user/miniconda3/envs/telepath3/lib/python3.8/site-packages/vaex/dataframe.py", line 3859, in trim
        df = self if inplace else self.copy()
      File "/home/user/miniconda3/envs/telepath3/lib/python3.8/site-packages/vaex/dataframe.py", line 5036, in copy
        df.add_column(name, column, dtype=self._dtypes_override.get(name))
      File "/home/user/miniconda3/envs/telepath3/lib/python3.8/site-packages/vaex/dataframe.py", line 6053, in add_column
        super(DataFrameArrays, self).add_column(name, data, dtype=dtype)
      File "/home/user/miniconda3/envs/telepath3/lib/python3.8/site-packages/vaex/dataframe.py", line 2942, in add_column
        raise ValueError("array is of length %s, while the length of the DataFrame is %s" % (len(ar), self.length_original()))
      ValueError: array is of length 274734, while the length of the DataFrame is 2
    '''

    res = r.sample(2)
    print(res)


def _test_vx_plot_client():
    r = ext_replay.VxBuffer(replay_csv_file_client)
    r.sample_strategy='rand'
    r.set_spec(replay_csv_file_client_spec)
    r.read_from_location()
    r.traj_order()
    r.compute_virtual_cols()

    dir = '/path/to/telepath/'
    f=None
    f=['dow','dow_cos','dow_sin']
    r.plots_2d( dir=dir, fields=f, forbid_fields=['study_arm_id','prediction_for_date'])
    r.plots_1d( dir=dir, fields=['dow'], forbid_fields=['study_arm_id','prediction_for_date'])
