from telepath_worker.chunk_sampler import ChunkSampler


def test_chunksampler_basic():
    cs = ChunkSampler(max_index=9,chunk_size=4,batch_size=2,chunk_resample_rate=1.0)

    for i in range(0,20):
        sp = cs.next_sample_params()
        print( sp )
    assert sp['chunk_start_index']==4
    assert sp['chunk_end_index']==8
    assert sp['epochs']==4
    assert sp['samples_per_chunk']==2.0

if __name__=="__main__":
    test_chunksampler_basic()

