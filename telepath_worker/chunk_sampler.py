
class ChunkSampler:

    batch_size = 0

    chunk_start_index = 0
    chunk_size = 10000
    samples_from_chunk = 0

    max_index = None
    epochs = 0
    chunk_resample_rate = 1.0

    def __init__(self,  max_index, chunk_size=10000, batch_size=32, chunk_resample_rate=0.1 ):
        self.max_index = max_index
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.chunk_resample_rate = chunk_resample_rate  # higher than 1.0 to reuse chunks more, lower for less
        self.reset()

    def reset(self):
        self.epochs=0
        self.samples_from_chunk=0
        self.samples_per_chunk=self.chunk_resample_rate*self.chunk_size//self.batch_size


    def next_sample_params(self):

        self.samples_from_chunk += 1

        ret = {'samples_from_chunk':self.samples_from_chunk, 'chunk_start_index':self.chunk_start_index, 'chunk_end_index':self.chunk_start_index + self.chunk_size,
                'epochs':self.epochs, 'samples_per_chunk': self.samples_per_chunk }

        if self.samples_from_chunk >= self.samples_per_chunk:
            self.chunk_start_index += self.chunk_size
            self.samples_from_chunk = 0

        if self.chunk_start_index + self.chunk_size >= self.max_index:
            self.chunk_start_index = 0
            self.epochs += 1

        return ret


if __name__=="__main__":
    cs = ChunkSampler(max_index=11,chunk_size=4,batch_size=2)

    for i in range(0,10):
        print( cs.next_sample_params() )
