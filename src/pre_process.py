from os.path import join
import os
import pandas as pd
from pandasql import sqldf
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from pandas import DataFrame
from tqdm import tqdm
import numpy as np


def load_binary(binary_dir = 'binary'):
    """
    loads the binary data.
    """
    df_active_days = pd.read_pickle(os.path.join(binary_dir, 'active_days.pkl'))
    df_cycles = pd.read_pickle(os.path.join(binary_dir, 'cycles.pkl'))
    df_users = pd.read_pickle(os.path.join(binary_dir, 'users.pkl'))
    df_tracking = pd.read_pickle(os.path.join(binary_dir, 'tracking.pkl'))
    df_labels = pd.read_pickle(os.path.join(binary_dir, 'labels.pkl'))

    return {'users': df_users,
            'cycles': df_cycles,
            'active_days': df_active_days,
            'tracking': df_tracking,
            'labels': df_labels}


def process_level1(tracking, cycles):
    """
    Pivot table and put symptoms on columns to get a matrix.
    Then convert the day_in_cycle into:
    - proportionate value to the cycle
    - distance from the end of the cycle
    - distance from the end of the period
    These values can be usefull later to map symptoms of
    different cycles together. It might also help mapping
    different cycles of different people together.
    
    The very helpful thing to map different cycles together (of the same id) 
    is to order them with respect to the ovulation day; the ovulation day becomes 
    then the origin time for predictions. Any symptom can appear some day before or after
    the ovulation day (see comment).
    
    """
    val = tracking[['user_id', 'cycle_id', 'day_in_cycle', 'date', 'symptom']]\
          .pivot_table(index= ['user_id', 'cycle_id', 'day_in_cycle', 'date'],
                       columns = ['symptom'], aggfunc=lambda x: 1).\
                       reset_index()

    val = sqldf("""select t1.*,
                   c1.cycle_length, c1.expected_cycle_length, c1.period_length,
                   t1.day_in_cycle / c1.cycle_length as proportionate,
                   c1.cycle_length - t1.day_in_cycle as inverse,
                   t1.day_in_cycle - c1.period_length as period_removed
                   from val as t1 join cycles as c1 on t1.user_id = c1.user_id
                   and t1.cycle_id = c1.cycle_id""",
                {'val': val, 'cycles': cycles})
    return val



def process_explode(tracking, cycles):
    """
    Pivot table and put symptoms on columns to get a matrix.

    Then explode missing dates

    Then convert the day_in_cycle into:
    - proportionate value to the cycle
    - distance from the end of the cycle
    - distance from the end of the period
    These values can be usefull later to map symptoms of
    different cycles together. It might also help mapping
    different cycles of different people together.
    """

    tracking_pivot = tracking[['user_id', 'cycle_id', 'day_in_cycle', 'date', 'symptom']] \
        .pivot_table(index=['user_id', 'cycle_id', 'day_in_cycle', 'date'],
                     columns=['symptom'], aggfunc=lambda x: 1). \
        reset_index()

    tracking_pivot["date"] = pd.to_datetime(tracking_pivot["date"])
    cycles["cycle_start"] = pd.to_datetime(cycles["cycle_start"])

    tracking_pivot = tracking_pivot.sort_values(by=["user_id", "date"], ascending=[False, True])
    tracking_pivot.reset_index(drop=True)

    full_cycles = []

    for i, cycle in tqdm(cycles.iterrows()):
        cycle_start = cycle["cycle_start"]
        cycle_length = cycle["cycle_length"]
        cycle_end = cycle_start + pd.DateOffset(cycle_length - 1)
        idx = pd.date_range(cycle_start, cycle_end, freq="D")
        full_df = DataFrame(idx, columns=["date"])

        full_df["user_id"] = cycle["user_id"]
        full_df["cycle_id"] = cycle["cycle_id"]
        full_df["period_length"] = cycle["period_length"]
        full_df["cycle_length"] = cycle_length
        full_df["expected_cycle_length"] = cycle["expected_cycle_length"]
        full_df["day_in_cycle"] = range(1, int(cycle_length) + 1, 1)
        full_df["proportionate"] = full_df["day_in_cycle"] / cycle_length
        full_df["inverse"] = cycle_length - full_df["day_in_cycle"]
        full_df["period_removed"] = full_df["day_in_cycle"] - cycle["period_length"]

        full_cycles.append(full_df)

    full_cycles = pd.concat(full_cycles)
    full_tracking_pivot = pd.merge(full_cycles, tracking_pivot, how='left',
                                   on=['user_id', 'cycle_id', "day_in_cycle", "date"])

    return full_tracking_pivot

def convert_to_X(val, users):
    a = val.groupby(('user_id', 'category', 'symptom', 'inverse_proportionate')).count().reset_index()

    a = a[['user_id', 'symptom', 'inverse_proportionate', 'cycle_id']]
    a.columns = ['user_id', 'symptom', 'inverse_proportionate', 'cnt']

    indexed_df = a.set_index(['user_id', 'inverse_proportionate', 'symptom'])
    for i in range(2):
        indexed_df = indexed_df.unstack(level=-1)
    indexed_df = indexed_df.reset_index()

    indexed_df.rename(columns={('user_id', '', ''): 'user_id'}, inplace=True)

    # Rename user_id column
    cols = list(indexed_df.columns)
    cols[0] = 'user_id'
    indexed_df.columns = cols

    indexed_df = pd.merge(pd.DataFrame(users['user_id']), indexed_df, how='left', on='user_id')

    return indexed_df

def process_level2(data: dict):
    min_cycle = pd.DataFrame(data['tracking'].groupby('user_id').cycle_id.min()).reset_index()
    min_cycle.columns = ['user_id', 'min_c']

    tracking = data['tracking']
    cycles = data['cycles']
    users = data['users']

    df = pd.merge(tracking, cycles, on=['user_id', 'cycle_id'])

    CYCLE_LEN = 29

    df['proportionate'] = df.day_in_cycle / df.cycle_length
    df['proportionate'] = df['proportionate'].astype(int)

    df['inverse'] = df.cycle_length - df.day_in_cycle
    df['inverse'] = df['inverse'].astype(int)

    df['inverse_proportionate'] = ((df.cycle_length - df.day_in_cycle) / df.cycle_length) * CYCLE_LEN
    df['inverse_proportionate'] = df['inverse_proportionate'].astype(int)

    v1 = pd.merge(df, min_cycle, on='user_id')
    vY = v1[v1.cycle_id==v1.min_c]

    Y = convert_to_X(vY, users)
    symptoms = ['happy', 'pms', 'sad', 'sensitive_emotion', 'energized', 'exhausted',
                'high_energy', 'low_energy', 'cramps', 'headache', 'ovulation_pain',
                'tender_breasts', 'acne_skin', 'good_skin', 'oily_skin', 'dry_skin']

    # Select only Y symptoms
    cols = list(Y.columns.values)
    cols = [x for x in cols if x[1] in symptoms]
    Y = Y[cols]

    # X
    vX = v1[v1.cycle_id != v1.min_c]
    X = convert_to_X(vX, users)
    X_all = convert_to_X(v1, users)


    assert X.shape[0] == Y.shape[0], "shape of X and Y does not agree"
    assert X.shape[0] == X_all.shape[0], "shape of X_all and X does not agree"

    return {'X': X,
            'Y': Y,
            'X_all': X_all}
    
