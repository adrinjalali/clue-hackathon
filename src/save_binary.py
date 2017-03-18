import sys
from os.path import join
import os
import pandas as pd


def save_binary(data_dir,
                binary_dir = 'binary',
                active_days = 'active_days.csv',
                cycles = 'cycles.csv',
                tracking = 'tracking.csv',
                users = 'users.csv'):
    """
    loads the data from the csv files
    then pickles them.
    """

    df_active_days = pd.read_csv(join(data_dir, active_days))
    df_cycles = pd.read_csv(join(data_dir, cycles))
    df_users = pd.read_csv(join(data_dir, users))
    df_tracking = pd.read_csv(join(data_dir, tracking))
    os.makedirs(binary_dir, exist_ok=True)
    df_active_days.to_pickle(os.path.join(binary_dir, 'active_days.pkl'))
    df_cycles.to_pickle(os.path.join(binary_dir, 'cycles.pkl'))
    df_users.to_pickle(os.path.join(binary_dir, 'users.pkl'))
    df_tracking.to_pickle(os.path.join(binary_dir, 'tracking.pkl'))


if __name__ == '__main__':
    data_fname = sys.argv[-1]
    save_binary(data_fname)
