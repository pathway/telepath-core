import os
from multiprocessing import Pool
import sys
import time

def system_run_process(final_cmd):
    debug = True
    if debug:
        print('[system_run_process] ' + final_cmd)
    os.system(final_cmd)  # blocks
    print('[system_run_process] done')


def multiproc_blocking_launch_nprocs(cmd_str, process_count=1):
    final_cmd_strings = []
    for port in range(0, process_count):
        final_cmd_strings.append('sleep '+str(port)+';  '+ cmd_str)

    print(final_cmd_strings)

    # this breaks very cleanly :)
    multiproc_launch_blocking(final_cmd_strings)


def multiproc_launch_blocking(final_cmd_strings,poolname='bmps_pool'):
    ''' input: a bunch of command strings.
    runs them all in parallel
    '''
    print('final_cmd_strings:',final_cmd_strings)
    pool = Pool(processes=len(final_cmd_strings))
    try:
        print('[{}] multiprocessing.pool.map begin...'.format(poolname))
        pool.map(system_run_process, final_cmd_strings) #blocks
        print('[{}] multiprocessing.pool.map ok'.format(poolname))
    except:
        print('[{}] multiprocessing.pool.close begin...'.format(poolname))
        pool.close()
        print('[{}] multiprocessing.pool.close ok'.format(poolname))
        pool.join()
        print('[{}] multiprocessing.pool.join ok'.format(poolname))

def _test():
    multiproc_launch_blocking(['sleep 4;echo "hi"','sleep 10;echo "bye"'])

if __name__=="__main__":
    print('Command line:', sys.argv)     # unchanged

    next = sys.argv.pop(0) # script name
    next = sys.argv.pop(0)
    procs = int(next)

    rest = " ".join(sys.argv)
    multiproc_blocking_launch_nprocs( rest, procs)
    #_test()

