import ext_replay
#import test.

replay_csv_file_small='test/test_data/replay_describe.csv'
replay_csv_file_huge='test/test_data/generated_tiny_cartpole.csv'  # see generate_replay.py
#replay_csv_file_huge='test/test_data/generated.csv'  # see generate_replay.py


r = ext_replay.PdPlainBuffer(replay_csv_file_huge)

r.chunksize = 6
r.reward = 'rew'
r.done = 'done'
r.action = 'action'

# r.read_from_location()
# print(r)
batch_size = 3
offset = 0
res = r.sample(batch_size)
print(res)
maxx=res.observations.max().item()
assert maxx <= r.chunksize
assert res.observations.min().item() >= 0

r.get_col_subsets()
print( r.subs)


offset += batch_size
res = r.sample(batch_size)
print(res)
assert res.observations.max().item() <= r.chunksize
assert res.observations.min().item() >= 0


offset += batch_size
res = r.sample(batch_size)
print(res)
minn = res.observations.min().item()
assert res.observations.max().item() <= r.chunksize * 2
assert minn >= r.chunksize
# pdb.set_trace()


offset += batch_size
res = r.sample(batch_size)
print(res)
minn = res.observations.min().item()


res=r.buffer_stats(batch_size=3)
print(res)

fn=replay_csv_file_huge+'.parquet'
print("writing "+fn)
r.export_format(fn)
#r.df.to_parquet(fn)
