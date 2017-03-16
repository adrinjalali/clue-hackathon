from os.path import join
import os
import pandas as pd
from pandasql import sqldf
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

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
    os.makedirs('binary', exist_ok=True)
    df_active_days.to_pickle('binary/active_days.pkl')
    df_cycles.to_pickle('binary/cycles.pkl')
    df_users.to_pickle('binary/users.pkl')
    df_tracking.to_pickle('binary/tracking.pkl')
    

def load_binary():
    """
    loads the binary data.
    """
    df_active_days = pd.read_pickle('binary/active_days.pkl')
    df_cycles = pd.read_pickle('binary/cycles.pkl')
    df_users = pd.read_pickle('binary/users.pkl')
    df_tracking = pd.read_pickle('binary/tracking.pkl')

    return {'users': df_users,
            'cycles': df_cycles,
            'active_days': df_active_days,
            'tracking': df_tracking}
    

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

def process_level2(l1_tracking, active_days):
    """
    WIP
    """
    pysqldf = lambda q: sqldf(q, globals())

    data = load_binary()
    users, cycles, active_days, tracking = data['users'], data['cycles'], data['active_days'], data['tracking']
    l1_tracking = process_level1(tracking, cycles)


    for user in users.user_id:
        a = l1_tracking[l1_tracking.user_id == user]
        b = active_days[active_days.user_id == user]
        if len(a) != len(b):
            print(user)
            break

        
    symptom = 'unprotected_sex'
    y = a[symptom].values
    y[np.isnan(y)] = 0
    x = np.array(list(range(len(a))))
    xx = np.linspace(x.min(),x.max(), len(x))
    itp = interp1d(x,y, kind='linear')
    window_size, poly_order = 5, 3
    yy_sg = savgol_filter(itp(xx), window_size, poly_order)
    print(np.vstack((yy_sg, y)).transpose())

    """
    map things to [0,1], by connecting cycles to tracking,
    group by day in cycle, take max or sum.
    """
