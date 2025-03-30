import json
from subprocess import Popen
import os, argparse

def script_energy(script: str, args: list=[]):
    '''
    runs a python script with perf stat energy-cores and args 
    '''
    # bench the energy costs of the test target
    p1 = Popen(['perf', 'stat', '--event', 'energy-cores', '-j', '-o', 'tmp.json', 'python3'] + [script] + args)
    err = p1.wait()

    # data is now in tmp.json
    with open('tmp.json', 'r') as f:
        next(f) # discard the comment header perf adds
        data = json.load(f)
    os.remove('tmp.json')

    return data['counter-value']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--n_avg', type=int, default=10, help='number of tests to average over')
    
    args = parser.parse_args()

    energies = []
    for i in range(args.n_avg):
        energies.append(float(script_energy('linear.py', args=['-e', '3'])))
        print(f'{i}: {energies[i]:.2f}')

    print(f'average: {sum(energies)/args.n_avg:.2f}J')
