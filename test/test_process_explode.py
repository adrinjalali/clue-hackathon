from os.path import join
import os
import pandas as pd
from pandasql import sqldf
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from pandas import DataFrame
from tqdm import tqdm


from src.pre_process import load_binary
from src.pre_process import process_explode


def test_load_binary():
    data = load_binary()
    users, cycles, active_days, tracking = data['users'], data['cycles'], data['active_days'], data['tracking']
    print(len(users))
    assert(len(users)>3000)

def test_process_explode():
    data = load_binary()
    users, cycles, active_days, tracking = data['users'], data['cycles'], data['active_days'], data['tracking']
    print("Going over all cycles ~20k:")
    fc = process_explode(tracking, cycles)
    print(len(fc))
    assert(len(fc)>600000)


if __name__ == "__main__":
    test_load_binary()
    test_process_explode()