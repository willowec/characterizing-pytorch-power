import json
from subprocess import Popen
import os, argparse
import sqlite3
import time


def args_dict_to_list(args_dict: dict):
    '''
    converts a dictionary of command line args e.g.
        {'-e': 3, '-nl': 10, '-v': ''}
    to a list passable to Popen like
        ['-e', '3', '-nl', '10', '-v', '']
    '''
    args_list = []
    for k, v in args_dict.items():
        args_list.append(str(k))
        args_list.append(str(v))
    return args_list


def script_energy(script: str, args_dict: dict={}):
    '''
    runs a python script with perf stat energy-pkg and args and returns energy, runtime
    '''

    args_list = args_dict_to_list(args_dict)
    
    # bench the energy costs of the test target
    tstart = time.time()
    p1 = Popen(['perf', 'stat', '--event', 'energy-pkg', '-j', '-o', 'tmp.json', 'python3'] + [script] + args_list)
    err = p1.wait()
    tend = time.time()

    # data is now in tmp.json
    with open('tmp.json', 'r') as f:
        next(f) # discard the comment header perf adds
        data = json.load(f)
    os.remove('tmp.json')

    return float(data['counter-value']), tend - tstart


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-n', '--n_average', type=int, default=5, help="num repetitions per run")

    args = parser.parse_args()

    start_time = time.time()

    # open database
    con = sqlite3.connect(f'{os.uname().nodename}_energy.db')
    cur = con.cursor()

    # create linear table
    cur.execute('CREATE TABLE IF NOT EXISTS linear(data_size INTEGER, epochs INTEGER, batch_size INTEGER, n_layers INTEGER, hidden_size INTEGER, energy REAL, traintime REAL)')

    # define iterations for each thingy. start, stop, step
    L_DATA_SIZE = [128, 1024, 128]
    L_EPOCHS = [10, 100, 10]
    L_BATCH_SIZE = [32, 512, 32]
    L_N_LAYERS = [1, 50, 1]
    L_HIDDEN_SIZE = [16, 256, 16]

    for data_size in range(*L_DATA_SIZE):
        for epochs in range(*L_EPOCHS):
            for batch_size in range (*L_BATCH_SIZE):
                for n_layers in range (*L_N_LAYERS):
                    for hidden_size in range(*L_HIDDEN_SIZE):
                        linear_args = {'-e': epochs, '-nl': n_layers, 
                                       '-ds': data_size, '-hs': hidden_size,
                                       '-bs': batch_size}

                        for i in range(args.n_average):
                            energy, traintime = script_energy('linear.py', linear_args)

                            print(f'linear {time.time()-start_time:.3e}s:\t{data_size=} {epochs=} {batch_size=} {n_layers=} {hidden_size=}: {energy=:.2f}J {traintime=:.2f}s')

                            cur.execute('INSERT INTO linear VALUES(?, ?, ?, ?, ?, ?, ?)',
                                        [data_size, epochs, batch_size, n_layers,
                                        hidden_size, energy, traintime])
                            con.commit()


    # create cnn table
    cur.execute('CREATE TABLE IF NOT EXISTS cnn(data_size INTEGER, epochs INTEGER, batch_size INTEGER, n_layers INTEGER, channels INTEGER, k_size INTEGER, stride INTEGER, side_length INTEGER, energy REAL, traintime REAL)')

    # define iterations for each thingy. start, stop, step
    C_DATA_SIZE = [128, 1024, 128]
    C_EPOCHS = [10, 100, 10]
    C_BATCH_SIZE = [32, 512, 32]
    C_N_LAYERS = [1, 50, 1]
    C_CHANNELS = [16, 128, 16]
    C_K_SIZE = [3, 9, 1]
    C_STRIDE = [1, 4, 1]
    C_SIDE_LENGTH = [16, 128, 16]

    for data_size in range(*C_DATA_SIZE):
        for epochs in range(*C_EPOCHS):
            for batch_size in range (*C_BATCH_SIZE):
                for n_layers in range (*C_N_LAYERS):
                    for channels in range(*C_CHANNELS):
                        for k_size in range (*C_K_SIZE):
                            for stride in range(*C_STRIDE):
                                for side_length in range(*C_SIDE_LENGTH):
                                    cnn_args = {'-e': epochs, 
                                                   '-nl': n_layers, 
                                                   '-ds': data_size, 
                                                   '-cs': channels,
                                                   '-bs': batch_size,
                                                   '-st': stride,
                                                   '-sl': side_length,
                                                   '-ks': k_size,
                                                   }
                                    
                                    for i in range(args.n_average):
                                        energy, traintime = script_energy('cnn.py', cnn_args)

                                        print(f'cnn {time.time()-start_time:.3e}s:\t{data_size=} {epochs=} {batch_size=} {n_layers=} {channels=} {k_size=} {stride=} {side_length=}: {energy=:.2f}J {traintime=:.2f}s')

                                        cur.execute('INSERT INTO cnn VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                                    [data_size, epochs, batch_size, n_layers,
                                                    channels, k_size, stride, side_length, 
                                                    energy, traintime])
                                        con.commit()
