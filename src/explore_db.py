DESC = """
Tool for exploring the energy databases
"""

LINEAR_TABLE='linear'
CNN_TABLE='cnn'
TABLES = [LINEAR_TABLE, CNN_TABLE]

ENERGY_COL='energy'
TRAIN_TIME_COL='traintime'

import argparse, sqlite3, random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def get_columns(cur: sqlite3.Cursor, table: str):
    """
    Returns the column names, excepting the energy and train times
    """
    return [fields[1] for fields in cur.execute(f"PRAGMA table_info({table})").fetchall()][:-2]


def get_uniques(cur: sqlite3.Cursor, table: str):
    """
    Gets the unique row combinations and return as list of row tuples
    """
    uniques = {}
    columns = get_columns(cur, table)

    cur.execute(f"SELECT DISTINCT {','.join(columns)} FROM {args.table}")
    return cur.fetchall()


def select_random_fixed_from_unique(uniques: list, table: str):
    """
    Selects a random fixed column value for each column from unique values and returns a dict
    """
    fixed = {}

    combo = random.choice(uniques)
    for i, c in enumerate(get_columns(cur, table)):
        fixed[c] = combo[i]

    return fixed


def select_longest_fixed_from_unique(cur: sqlite3.Cursor, col: str, uniques: list, table: str):
    """
    Gets the counts for every unique set varying over col, returns set with highest count
    """
    columns = get_columns(cur, table)
    longest_count = 0
    longest_fixed = {}
    for combo in uniques:
        fixed = {}
        for i, c in enumerate(get_columns(cur, table)):
            fixed[c] = combo[i]
        count = get_column_along(cur, col, fixed, table).shape[0]
        if count > longest_count:
            longest_count = count
            longest_fixed = fixed

    return longest_fixed


def get_column_along(cur: sqlite3.Cursor, col: str, fixed: dict, table: str):
    """
    Returns the value for one column along a fixed axis from all other columns
    """
    query = f"SELECT * FROM {table} WHERE "
    for k, v in fixed.items():
        if k != col: # fix everything except our target col
            query += f"{k}={v} AND "

    query = query[:-4] # cut off the last AND

    cur.execute(query)
    return np.asarray(cur.fetchall())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dbfile', type=Path, help='file path to the sqlite3 database')
    parser.add_argument('-c', '--col', type=str, default=None, help='the column you want to explore')
    parser.add_argument('-t', '--table', type=str, default=LINEAR_TABLE, choices=TABLES, help='the table you want to explore')
    parser.add_argument('-pc', '--print_columns', action='store_true', help='print the columns and exit')

    args = parser.parse_args()

    # connect to the DB
    con = sqlite3.connect(args.dbfile)
    cur = con.cursor()

    if args.print_columns:
        print(get_columns(cur, args.table))
        quit()

    uniques = get_uniques(cur, args.table)
    #fixed = select_random_fixed_from_unique(uniques, args.table)
    fixed = select_longest_fixed_from_unique(cur, args.col, uniques, args.table)



    if args.col:
        columns = get_columns(cur, args.table)
        along = get_column_along(cur, args.col, fixed, args.table)
        target = along[:, columns.index(args.col)]
        energy = along[:, -2]
        traintime = along[:, -1]
    
        f, ax = plt.subplots(2)
        ax[0].scatter(target, energy, color='red')
        ax[0].set_ylabel('energy (J)')

        ax[1].scatter(target, traintime, color='blue')
        ax[1].set_ylabel('train time (s)')

        f.suptitle(f"Column {args.col} along {[f'{k}={v}' for k, v in fixed.items() if k != args.col]}")
        f.supxlabel(f"{args.col}")
        plt.show()