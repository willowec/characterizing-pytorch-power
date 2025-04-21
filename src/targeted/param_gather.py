'''
Targeted exploration of parameters.
Fixes all parameters to default values and then varies target over range
Stores into json for analysis
    # of model parameters
    energy used
    time training
'''

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from subprocess import Popen, PIPE
import json, time, os

OUT_DIR = Path('./out')

# model scripts
LINEAR_SCRIPT = Path('../general/linear.py')
CNN_SCRIPT = Path('../general/cnn.py')
MODEL_SCRIPTS = [LINEAR_SCRIPT, CNN_SCRIPT]

# fixed parameter defaults, min, max, step
PARAMS_LINEAR = {
    '--data_size':      {'fixed': 512,  'range': [64, 4096, 64]}, 
    '--input_size':     {'fixed': 128,  'range': [16, 512, 16]},
    '--output_size':    {'fixed': 128,  'range': [16, 512, 16]},
    '--batch_size':     {'fixed': 512,  'range': [32, 2048, 32]},
    '--epochs':         {'fixed': 10,   'range': [2, 100, 1]},
}

PARAMS_CNN = {
    '--data_size':      {'fixed': 2048, 'range': [64, 4096, 64]}, 
    '--batch_size':     {'fixed': 512,  'range': [32, 2048, 32]},
    '--epochs':         {'fixed': 10,   'range': [2, 100, 1]},
    '--in_channels':    {'fixed': 16,   'range': [2, 64, 2]},
    '--out_channels':   {'fixed': 16,   'range': [2, 64, 2]},
    '--k_size':         {'fixed': 3,    'range': [3, 15, 3]},
    '--stride':         {'fixed': 1,    'range': [1, 8, 1]},
    '--side_length':    {'fixed': 32,   'range': [8, 256, 8]},
}

MODEL_PARAMS = [PARAMS_LINEAR, PARAMS_CNN]

N_AVERAGE=10    # number of times to repeat each test


def script_energy(script: Path, args_dict: dict={}):
    '''
    runs a python script with perf stat energy-pkg and args and returns energy, runtime, and model params
    '''

    args_list = []
    for k, v in args_dict.items():
        args_list.append(str(k))
        args_list.append(str(v))
    
    # bench the energy costs of the test target
    t_start = time.time()
    p1 = Popen(['perf', 'stat', '--event', 'energy-pkg', '-j', '-o', 'tmp.json', 'python3'] + [str(script)] + args_list, stdout=PIPE)
    out, err = p1.communicate()
    
    n_model_params = out.decode()
    tend = time.time()

    # data is now in tmp.json
    with open('tmp.json', 'r') as f:
        for line in f.readlines():
            if line.startswith('{'):
                data = json.loads(line)
                break
    os.remove('tmp.json')

    return float(data['counter-value']), tend - t_start, int(n_model_params)


def explore_param(model_script: Path, params: dict, target: str):
    '''
    Explores the target param in params for the model_script
    '''
    # build fixed params dict
    fixed_params = {}
    for k, v in params.items():
        fixed_params[k] = v['fixed']

    param_vals = []
    energies = []
    train_times = []
    model_n_params = []

    # now iterate over target parameter space
    for i in range(*params[target]['range']):
        fixed_params[target] = i
        for j in range(N_AVERAGE):
            e, t, p = script_energy(model_script, fixed_params)
            param_vals.append(i)
            energies.append(e)
            train_times.append(t)
            model_n_params.append(p)

    return param_vals, energies, train_times, model_n_params



if __name__ == '__main__':
    for script, params in zip(MODEL_SCRIPTS, MODEL_PARAMS):
        for param in params.keys():

            print(f'Running {str(script.name)}:\t{param=}...\t', end='', flush=True)

            param_vals, energies, train_times, model_n_params = explore_param(
                script, params, param)
    
            with open(OUT_DIR.joinpath(f'{script.stem}-{param.replace("-", "")}.json'), 'w+') as f:
                data = {'fixed': {}}
                for k, v in params.items():
                    data['fixed'][k] = v

                data['param'] = param
                data['param_vals'] = param_vals
                data['energies'] = energies
                data['train_times'] = train_times
                data['model_n_params'] = model_n_params

                json.dump(data, f)

            print(f'finished')