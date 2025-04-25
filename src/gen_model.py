'''
Generate the prediction models for each layer type
'''

import argparse
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=Path, help='path to the layer data')

    args = parser.parse_args()

    df = pd.read_csv(args.data_file)

    X = df.drop(columns=['for_energy', 'for_time', 'back_energy', 'back_time'])
    y = df['for_energy']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 1)

    scaler = StandardScaler()
    X_train_scaler = scaler.fit_transform(X_train)
    X_test_scaler = scaler.transform(X_test)

    lin = LinearRegression()

    poly = PolynomialFeatures(degree=3)
    X_poly_train = poly.fit_transform(X_train_scaler)
    print(X_poly_train.shape)
    X_test_poly = poly.transform(X_test_scaler)
    poly.fit(X_poly_train, y_train)
    lin.fit(X_poly_train, y_train)

    y_pred = lin.predict(X_test_poly)

    print(mean_absolute_error(y_test, y_pred))

    y_pred_train = lin.predict(X_poly_train)
    print(mean_absolute_error(y_train, y_pred_train))