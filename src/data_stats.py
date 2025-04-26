'''
get some stats of a data run
'''

import argparse
import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.data_file)
    for_time = np.sum(df['for_time']) / 1e6 # convert to seconds
    back_time = np.sum(df['back_time']) / 1e6 # convert to seconds

    print(f'Total time: {for_time+back_time:3f}s.')