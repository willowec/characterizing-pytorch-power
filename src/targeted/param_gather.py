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
from subprocess import Popen
import json, time, os

OUT_DIR = Path('./out')

# model scripts
LINEAR_SCRIPT = Path('../cnn.py')
CNN_SCRIPT = Path('../cnn.py')
MODEL_SCRIPTS = [CNN_SCRIPT, LINEAR_SCRIPT]

# fixed parameter defaults, min, max, step
PARAMS_LINEAR = { # TODO shouldn't I have input/output sizes?
    '--epochs':          {'fixed': 10,   'range': [2, 100, 1]},
    '--n_layers':       {'fixed': 1,    'range': [1, 50, 1]}, 
    '--data_size':      {'fixed': 2048, 'range': [64, 4096, 64]}, 
    '--hidden_size':    {'fixed': 128,  'range': [16, 512, 16]},
    '--batch_size':     {'fixed': 512,  'range': [32, 2048, 32]}
}

PARAMS_CNN = {
    '--data_size':      {'fixed': 2048, 'range': [64, 4096, 64]}, 
    '--n_layers':       {'fixed': 1,    'range': [1, 50, 1]}, 
    '--batch_size':     {'fixed': 512,  'range': [32, 2048, 32]},
    '--epochs':         {'fixed': 10,   'range': [2, 100, 1]},
    '--channels':       {'fixed': 16,   'range': [2, 64, 2]},
    '--k_size':         {'fixed': 3,   'range': [3, 15, 3]},
    '--stride':         {'fixed': 1,   'range': [1, 8, 1]},
    '--side_length':    {'fixed': 32,   'range': [8, 256, 8]},
}

N_AVERAGE=10    # number of times to repeat each test


def script_energy(script: str, args_dict: dict={}):
    '''
    runs a python script with perf stat energy-pkg and args and returns energy, runtime
    '''

    args_list = []
    for k, v in args_dict.items():
        args_list.append(k)
        args_list.append(v)
    
    # bench the energy costs of the test target
    t_start = time.time()
    p1 = Popen(['perf', 'stat', '--event', 'energy-pkg', '-j', '-o', 'tmp.json', 'python3'] + [script] + args_list)
    err = p1.wait()
    tend = time.time()

    # data is now in tmp.json
    with open('tmp.json', 'r') as f:
        for line in f.readlines():
            if line.startswith('{'):
                data = json.loads(line)
                break
    os.remove('tmp.json')

    return float(data['counter-value']), tend - t_start


def explore_param(model_script: Path, params: dict, target: str):
    '''
    Explores the target param in params for the model_script
    '''
    # build fixed params dict
    fixed_params = {}
    for k, v in params.items():
        fixed_params[k] = v['fixed']

    # now iterate over target parameter space
    for i in range(*params[target]['range']):
        fixed_params[target] = i
        for j in range(N_AVERAGE):
            e, t = script_energy(model_script, fixed_params)


if __name__ == '__main__':
    explore_param(LINEAR_SCRIPT, PARAMS_LINEAR, '--batch_size')