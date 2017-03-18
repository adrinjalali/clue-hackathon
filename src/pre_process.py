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

def convert_to_X(val):
    day_in_cycle = pd.DataFrame(np.array(range(29)), columns = ['day'])

    val.proportionate = val.proportionate.astype(int)
    val.inverse_proportionate = val.inverse_proportionate.astype(int)

    v2 = sqldf("""select t1.user_id, t1.category, t1.symptom, t1.inverse_proportionate as X, count(*) as cnt
                  from tbl as t1
                  group by user_id, category, symptom, inverse_proportionate
                  order by inverse_proportionate""",
               {'tbl': val})

    v3 = v2.pivot_table(index= ['user_id'],
                        columns = ['X', 'category', 'symptom'], aggfunc=lambda x: sum(x)).\
                        reset_index()
    return v3

def process_level2(data):
    min_cycle = sqldf("""select user_id, min(cycle_id) as min_c from tracking 
                         group by user_id""", data)
    
    val = sqldf("""select t1.*,
                   c1.cycle_length, c1.expected_cycle_length, c1.period_length,
                   (t1.day_in_cycle / c1.cycle_length) * 10 as proportionate,
                   c1.cycle_length - t1.day_in_cycle as inverse,
                   ((c1.cycle_length - t1.day_in_cycle) / c1.cycle_length) * 10 as inverse_proportionate,
                   t1.day_in_cycle - c1.period_length as period_removed
                   from val as t1 join cycles as c1 on t1.user_id = c1.user_id
                   and t1.cycle_id = c1.cycle_id""",
                {'val': data['tracking'], 'cycles': data['cycles']})

    v1 = sqldf("""select val.* from val
                  join min_cycle on val.user_id == min_cycle.user_id
                  where val.cycle_id == min_cycle.min_c""",
               {'val': val, 'min_cycle': min_cycle})
    Y = convert_to_X(v1)
    symptoms = ['happy', 'pms', 'sad', 'sensitive_emotion', 'energized', 'exhausted',
                'high_energy', 'low_energy', 'cramps', 'headache', 'ovulation_pain',
                'tender_breasts', 'acne_skin', 'good_skin', 'oily_skin', 'dry_skin']
    cols = list(Y.columns.values)
    cols = [x for x in cols if x[3] in symptoms]
    Y = Y[cols]
    
    v1 = sqldf("""select val.* from val
                  join min_cycle on val.user_id == min_cycle.user_id
                  where val.cycle_id != min_cycle.min_c""",
               {'val': val, 'min_cycle': min_cycle})
    X = convert_to_X(v1)
    
    v1 = sqldf("""select val.* from val
                  join min_cycle on val.user_id == min_cycle.user_id
                  where val.cycle_id != min_cycle.min_c""",
               {'val': val, 'min_cycle': min_cycle})
    X_all = convert_to_X(v1)

    return {'X': X,
            'Y': Y,
            'X_all': X_all}
    
