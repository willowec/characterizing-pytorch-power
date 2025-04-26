'''
Generate the prediction models for each layer type
'''

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.metrics import root_mean_squared_error


def train_model(X_train, y_train, degree):
    '''
    Trains a polynomial regression of order degree
    '''

    lin = LinearRegression()
    poly = PolynomialFeatures(degree=degree)

    X_poly_train = poly.fit_transform(X_train)
    
    poly.fit(X_poly_train, y_train)
    lin.fit(X_poly_train, y_train)

    return lin, poly


def pred_model(model, poly, X):
    '''
    Forward pass a linear regression model
    '''
    X_poly = poly.transform(X)
    return model.predict(X_poly)


def search_orders(X, y, train_split=.25, max_order=6):
    '''
    creates a model for each order and tests it to find the one with the lowest test error
    TODO since this seems to give inconsistent results maybe cross-validation would be better
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    errs = []
    for i in range(max_order):
        try:
            lin = train_model(X_train, y_train, i)
            
            y_pred_test = pred_model(lin, X_test, i)
            errs.append(root_mean_squared_error(y_test, y_pred_test))
        except ValueError as e:
            break # we probably tried an order that was too big 

    return np.argmin(errs), errs



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=Path, help='path to the layer data')
    parser.add_argument('--search', action='store_true', help='search for the best order for this problem')
    parser.add_argument('-d', '--degree', type=int, default=1, help='polynomial degree to use')
    parser.add_argument('-N', '--N_sets', type=int, default=1, help='the number fo times to randomize the train/test data and re-train')
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
    y = df['for_energy']

    if args.search:
        order, test_errs = search_orders(X, y)
        print(f'Best order: {order}. Test errors: {test_errs}')
        quit()

    for i in range(args.N_sets):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        scaler = StandardScaler()
        X_train_scaler = scaler.fit_transform(X_train)
        X_test_scaler = scaler.transform(X_test)

        lin = Lasso()

        poly = PolynomialFeatures(degree=args.degree)
        X_train_poly = poly.fit_transform(X_train_scaler)
        X_test_poly = poly.transform(X_test_scaler)

        poly.fit(X_train_poly, y_train)
        lin.fit(X_train_poly, y_train)

        y_pred_test = lin.predict(X_test_poly)
        y_pred_train = lin.predict(X_train_poly)

        test_err = root_mean_squared_error(y_test, y_pred_test)
        train_err = root_mean_squared_error(y_train, y_pred_train)

        print(f"{i}: {test_err=}\t{train_err=}")