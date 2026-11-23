import h5py
import tables
import numpy as np
from time import time

cdim, block_size, row_loops = 64, 4, 381854
vals = np.ones((block_size, cdim), dtype="float32")

if 0:
    # append rows
    print("PYTABLES: append rows: %d blocks with: %d rows" % (row_loops, block_size))
    print("=" * 32)
    f = tables.open_file("rowapp_test_tb.h5", "w")
    a = f.create_earray(f.root, "time_data", atom=tables.Float32Atom(), shape=(0, cdim))
    t1 = time()
    for i in range(row_loops):
        a.append(vals)
    tcre = round(time() - t1, 3)
    thcre = round(cdim * block_size * row_loops * 4 / (tcre * 1024 * 1024), 1)
    print("Time to append %d rows: %s sec (%s MB/s)" % (block_size * row_loops, tcre, thcre))
    print("=" * 32)
    chunkshape = a.chunkshape
    f.close()

if 0:
    import vaex as vx
    df = vx.from_csv('../telepath_worker/test/test_data/generated.csv')
    df.export_hdf5("generated.csv.hdf5", progress=True)


if 1:
    import h5py
    fn = 'generated.csv.hdf5'
    #fn = 'rowapp_test_tb.h5'
    f = h5py.File(fn, 'r')
    for item in f.attrs.keys():
        print(item + ":", f.attrs[item])
    thing = f['/']
    #print(thing)
    #print(thing[:].sum() )
    #mr = f['/entry/mr_scan/mr']
    #i00 = f['/entry/mr_scan/I00']
    #print("%s\t%s\t%s" % ("#", "mr", "I00"))
    #for i in range(len(mr)):
    #    print("%d\t%g\t%d" % (i, mr[i], i00[i]))
    f.close()

exit(1)

if 0:
    print("H5PY: append rows %d blocks with: %d rows" % (row_loops, block_size))
    print("=" * 32)
    f = h5py.File(name="rowapp_test_h5.h5",mode='w')
    a = f.create_dataset(name='time_data',shape=(0, cdim),
                         maxshape=(block_size*row_loops,cdim),
                         dtype='f',chunks=chunkshape)
    t1 = time()
    samplesWritten = 0
    for i in range(row_loops):
        a.resize(((i+1)*block_size, cdim))
        a[samplesWritten:samplesWritten+block_size] = vals
        samplesWritten += block_size
    tcre = round(time() - t1, 3)
    thcre = round(cdim * block_size * row_loops * 4 / (tcre * 1024 * 1024), 1)
    print("Time to append %d rows: %s sec (%s MB/s)" % (block_size * row_loops, tcre, thcre))
    print("=" * 32)
    f.close()
