import json
from subprocess import Popen
import os

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

    print(script_energy('linear.py'))
