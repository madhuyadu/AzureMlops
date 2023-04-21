# Import libraries

import argparse
import glob
import os

import pandas as pd

from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow

# define functions
def main(args):
    # TO DO: enable autologging

    mlflow.autolog()
    print("inside main function")
    print("autologging starts from here...")
    print("main code execution starts...")
    # read data
    df = get_csvs_df(args.training_data)
    print("input file is: ")
    print(df.head(3))

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    print("inside get_csvs_df function")
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# TO DO: add function to split data
def split_data(df):
    X = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values 
    y = df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    # data = {"train": {"X": X_train, "y": y_train},
    #         "test": {"X": X_test, "y": y_test}}
    print("train test split done")
    return X_train, X_test, y_train, y_test
    


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    print("Model training in progress..")
    LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)


def parse_args():
    print("inside parse_args function")
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("Code execution starts...")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
