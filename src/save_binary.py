import sys
from os.path import join
import os
import pandas as pd


def save_binary(data_dir,
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
    os.makedirs('../binary', exist_ok=True)
    df_active_days.to_pickle('../binary/active_days.pkl')
    df_cycles.to_pickle('../binary/cycles.pkl')
    df_users.to_pickle('../binary/users.pkl')
    df_tracking.to_pickle('../binary/tracking.pkl')


if __name__ == '__main__':
    data_fname = sys.argv[-1]
    save_binary(data_fname)
