import json
from subprocess import Popen
import os, argparse


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


def script_energy(script: str, n_avg=10, args_dict: dict={}):
    '''
    runs a python script with perf stat energy-cores and args 
    '''

    args_list = args_dict_to_list(args_dict)
    energies = []
    for _ in range(n_avg):
        # bench the energy costs of the test target
        p1 = Popen(['perf', 'stat', '--event', 'energy-cores', '-j', '-o', 'tmp.json', 'python3'] + [script] + args_list)
        err = p1.wait()

        # data is now in tmp.json
        with open('tmp.json', 'r') as f:
            next(f) # discard the comment header perf adds
            data = json.load(f)
        os.remove('tmp.json')
        energies.append(float(data['counter-value']))

    return sum(energies) / n_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--n_avg', type=int, default=10, help='number of tests to average over')
    
    args = parser.parse_args()


    energy = []
    num_layers = []
    for i in range(1, 100):
        linear_args = {'-e': 1, '-nl': i}
        energy.append(script_energy('linear.py', args.n_avg, linear_args))

        print(linear_args, energy[i-1])

    print(energy)
