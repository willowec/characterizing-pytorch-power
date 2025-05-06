'''
Generate the prediction models for each layer type
'''

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import pickle

import warnings

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
from sklearn.exceptions import ConvergenceWarning

# cause enbies really just do want to have fun so desparately
COLOR_GREEN = "\x1b[92m"
COLOR_RED = "\x1b[91m"
COLOR_RESET = "\x1b[0m"

RANDOM_STATE = 10

def train_model(X, y, lin, degree):
    '''
    Trains a polynomial regression of order degree on linear model lin
    '''
    lin = lin()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    X_train_scaler = scaler.fit_transform(X_train)
    X_test_scaler = scaler.transform(X_test)

    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_scaler, y_train)
    X_test_poly = poly.transform(X_test_scaler)

    lin.fit(X_train_poly, y_train)
    y_pred_test = lin.predict(X_test_poly)
    y_pred_train = lin.predict(X_train_poly)

    test_rmse = root_mean_squared_error(y_test, y_pred_test)
    train_rmse = root_mean_squared_error(y_train, y_pred_train)
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
    train_mape = mean_absolute_percentage_error(y_train, y_pred_train)

    return lin, poly, scaler, test_rmse, train_rmse, test_mape, train_mape


def search_models_orders(X, y, lins: list, degrees: int, n_avg: int=10) -> tuple:
    '''
    Searches the entire space to look for the best combination of model and polynomial degree
    returns the best model as a tuple of (linear model, degree, test err)
    '''
    best = (lins[0], degrees, None, None, np.inf) # the best model (arch, degree, poly, scaler, test err)

    for model in lins:
        for degree in range(1, degrees+1):
            test_errs = []
            train_errs = []

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

                for i in range(n_avg):
                    lin, poly, scaler, test_rmse, train_rmse, test_mape, train_mape = train_model(X.values, y.values, model, degree)
                    test_errs.append(test_rmse)
                    train_errs.append(train_rmse)

                test_err = np.mean(test_errs)
                train_err = np.mean(train_errs)

                if test_err < best[-1]:
                    # this test err was better than the previous best
                    best = (lin, degree, poly, scaler, test_err)

                print(f'Model {lin} with degree {degree}:\t{test_rmse=:.3f}\t{train_rmse=:.3f}\t{test_mape=:.2}%\t{train_mape=:.2}%')

    print(f'Best model: {best[0]} (input features {best[0].n_features_in_}) with degree {best[1]}:\t{COLOR_GREEN}test_rmse={best[-1]:.3e}{COLOR_RESET}')
    return best


def save_model(model: tuple, path: Path):
    '''
    Save a model to pickle file
    '''
    with open (path, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=Path, help='path to the layer data')
    parser.add_argument('--search', action='store_true', default=False, help='search for the best order for this problem')
    parser.add_argument('-d', '--degree', type=int, default=1, help='polynomial degree to use')
    parser.add_argument('-N', '--N_sets', type=int, default=1, help='the number of times to randomize the train/test data and re-train')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)

    args = parser.parse_args()

    # open data
    df = pd.read_csv(args.data_file)
    df = df.iloc[args.start:args.end]

    if args.end != -1:
        # debug print the last couple rows we are including
        print(df.iloc[-5:])

    # drop rows with negatives (no data should ever be less than or equal to 0)
    df = df[df['for_energy'] >= 0]
    df = df[df['back_energy'] >= 0]
    df = df[df['for_time'] >= 0]
    df = df[df['back_time'] >= 0]

    df['for_energy'] /= 1e6 # convert to Joules
    df['back_energy'] /= 1e6 # convert to Joules
    df['for_time'] /=1e6 # convert to seconds
    df['back_time'] /=1e6 # convert to seconds

    # extract x and y features from data
    X = df.drop(columns=['for_energy', 'for_time', 'back_energy', 'back_time'])
    y = df[['for_energy', 'for_time', 'back_energy', 'back_time']]

    if args.search:
        for column in y.columns:
            print(f'Generating model for {column}:')

            # gen model (make sure to pass the models uninitialized!)
            model = search_models_orders(X, y[column], [LinearRegression, Lasso, LassoCV], args.degree) 
            
            # save model
            model_dir = Path(f"out/models/{args.data_file.parts[1].split('-')[0]}/{column}")
            model_dir.mkdir(exist_ok=True, parents=True)
            save_model(model, model_dir.joinpath(args.data_file.stem + '.pickle'))
            
        quit()

    for i in range(args.N_sets):
        for column in y.columns:
            print(f'Generating model for {column}:')

            lin, poly, scaler, test_rmse, train_rmse, test_mape, train_mape = train_model(X.values, y[column].values, LassoCV, args.degree)
            model = (lin, args.degree, poly, scaler, test_rmse)

            print(f"{i}: {test_rmse=}\t{train_rmse=}\t{test_mape=:.2}%\t{train_mape=:.2}%")

            # save model
            model_dir = Path(f"out/models/{args.data_file.parts[1].split('-')[0]}/{column}")
            model_dir.mkdir(exist_ok=True, parents=True)
            save_model(model, model_dir.joinpath(args.data_file.stem + '.pickle'))