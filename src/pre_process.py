import os
import pandas as pd
import numpy as np
import csv


def load_binary(binary_dir='binary'):
    """Loads the binary data.
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


def preprocess_users(users):
    # Preparing countries to continent mapping
    reader = csv.reader(open('clustering/countries_mapping.csv', 'r'))
    d = {}
    for row in reader:
        d[row[1]] = row[0]

    # Preparing data
    df_users = users.apply(lambda x: x.fillna(x.median()) if np.issubdtype(x.dtype, np.number) else x, axis=0)
    df_users['continent'] = df_users.country.map(d).fillna("Oceania")
    df_users.continent = df_users.continent.apply(lambda x: 'Asia' if x == 'South Korea' else x)
    # df_u['phase'] = ((df_u.birthyear - df_u.birthyear.min())/(df_u.birthyear.max() - df_u.birthyear.min()))
    # df_u['phase'] = df_u.phase.apply(lambda x: 0 if x > 0.8 else 2 if x < 0.2 else 1)
    # df_u['adolescence'] = (((df_u.birthyear - df_u.birthyear.min())/(df_u.birthyear.max() - df_u.birthyear.min())) > 0.8) * 1
    # df_u['menopause'] = (((df_u.birthyear - df_u.birthyear.min())/(df_u.birthyear.max() - df_u.birthyear.min())) < 0.2) * 1
    df_users['age'] = 2017 - df_users.birthyear
    df_users['age_bracket'] = df_users.age.apply(lambda x: '<18' if x < 18 else '18-24' if x > 24 else '>24')
    df_users['first_havers'] = (2017 - df_users.birthyear).apply(lambda x: 1 if x < 15 else 0)
    df_users['menopause'] = (((df_users.birthyear - df_users.birthyear.min())/(df_users.birthyear.max() - df_users.birthyear.min())) < 0.2) * 1
    df_users['bmi'] = df_users.weight / ((df_users.height/100)**2)

    # Preparing the dataset to cluster
    df_users_adj = pd.concat([df_users, pd.get_dummies(df_users.continent), pd.get_dummies(df_users.age_bracket)], axis=1) \
                    .drop(['Oceania', 'country', 'platform', 'continent', \
                            'birthyear', 'weight', 'height','age','age_bracket'], axis = 1)

    return df_users_adj


def convert_to_X(val, users, active_days, day_transform):
    """
    This function converts the data from row format,
    into column format. The data is converted into the
    format which as only one row per user in this function.
    """
    a = val.groupby(('user_id', 'category', 'symptom', day_transform)).count().reset_index()
    c_count = val.groupby('user_id').cycle_id.nunique().reset_index()
    c_count.columns = ['user_id', 'cycle_count']
    a = a.merge(c_count, on = 'user_id').reset_index()
    a = a[['user_id', 'symptom', day_transform, 'cycle_id', 'cycle_count']]
    a.columns = ['user_id', 'symptom', day_transform, 'cnt', 'cycle_count']
    a['cnt'] = a['cnt'] / a['cycle_count']
    a = a[['user_id', 'symptom', day_transform, 'cnt']]

    indexed_df = a.set_index(['user_id', day_transform, 'symptom'])
    for i in range(2):
        indexed_df = indexed_df.unstack(level=-1)
    indexed_df = indexed_df.reset_index()

    indexed_df.rename(columns={('user_id', '', ''): 'user_id'}, inplace=True)

    # Rename user_id column
    cols = list(indexed_df.columns)
    cols[0] = 'user_id'
    indexed_df.columns = cols

    indexed_df = pd.merge(pd.DataFrame(users['user_id']), indexed_df, how='left', on='user_id')

    ad = active_days.groupby(('user_id', 'cycle_id')).count().reset_index().\
            groupby('user_id').median().reset_index()[['user_id', 'date']]
    ad.columns = ['user_id', 'active_days']
    indexed_df = pd.merge(indexed_df, ad, on='user_id').reset_index()

    # Gianluca
    #df_users_adj = preprocess_users(users)
    #indexed_df = pd.merge(df_users_adj, indexed_df, how='left', on='user_id')
    
    return indexed_df


def process_level2(data: dict, double_explode = False):
    """
    This function takes the raw data read from csv files
    in terms of a dictionary, and returns X and Y for
    model training, and X_all for the final prediction.
    """
    
    min_cycle = pd.DataFrame(data['tracking'].groupby('user_id').cycle_id.min()).reset_index()
    min_cycle.columns = ['user_id', 'min_c']

    tracking = data['tracking']
    cycles = data['cycles']
    users = data['users']
    active_days = data['active_days']

    df = pd.merge(tracking, cycles, on=['user_id', 'cycle_id'])

    CYCLE_LEN = 29

    df['proportionate'] = df.day_in_cycle / df.cycle_length * CYCLE_LEN
    df['proportionate'] = df['proportionate'].astype(int)

    df['inverse'] = df.cycle_length - df.day_in_cycle
    df['inverse'] = df['inverse'].astype(int)

    df['inverse_proportionate'] = ((df.cycle_length - df.day_in_cycle) / df.cycle_length) * CYCLE_LEN
    df['inverse_proportionate'] = df['inverse_proportionate'].astype(int)

    v1 = pd.merge(df, min_cycle, on='user_id')
    vY = v1[v1.cycle_id == v1.min_c]

    Y = convert_to_X(vY, users, active_days, 'proportionate')
    symptoms = ['happy', 'pms', 'sad', 'sensitive_emotion', 'energized', 'exhausted',
                'high_energy', 'low_energy', 'cramps', 'headache', 'ovulation_pain',
                'tender_breasts', 'acne_skin', 'good_skin', 'oily_skin', 'dry_skin']

    # Select only Y symptoms
    cols = list(Y.columns.values)
    cols = [x for x in cols if x[1] in symptoms]
    Y = Y[cols]

    # X
    if not double_explode:
        vX = v1[v1.cycle_id != v1.min_c]
        X = convert_to_X(vX, users, active_days, 'proportionate')
        X_all = convert_to_X(v1, users, active_days, 'proportionate')
    else:
        vX = v1[v1.cycle_id != v1.min_c]
        Xip = convert_to_X(vX, users, active_days, 'inverse_proportionate')
        Xip = Xip.set_index('user_id')
        Xp = convert_to_X(vX, users, active_days, 'proportionate')
        Xp = Xp.set_index('user_id')
        X = pd.concat([Xip, Xp], axis=1).reset_index()
        X_all_ip = convert_to_X(v1, users, active_days, 'inverse_proportionate')
        X_all_ip = X_all_ip.set_index('user_id')
        X_all_p = convert_to_X(v1, users, active_days, 'proportionate')
        X_all_p = X_all_p.set_index('user_id')
        X_all = pd.concat([X_all_ip, X_all_p], axis=1).reset_index()


    assert X.shape[0] == Y.shape[0], "shape of X and Y does not agree"
    assert X.shape[0] == X_all.shape[0], "shape of X_all and X does not agree"

    return {'X': X,
            'Y': Y,
            'X_all': X_all}
