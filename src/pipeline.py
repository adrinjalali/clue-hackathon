import numpy as np
import sys
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from os.path import join
from src.dump_results import dump

from src.pre_process import process_level2, load_binary


def run():
    data = load_binary()

    # (Opt: user clustering)

    # Extract features
    user_feat_matrix = process_level2(data)  # X

    del user_feat_matrix['X']['user_id']
    X = user_feat_matrix['X'].values
    X[np.isnan(X)] = 0
    Y = user_feat_matrix['Y']
    Y.fillna(0, inplace=True)
    del user_feat_matrix['X_all']['user_id']
    X_all = user_feat_matrix['X_all'].values
    X_all[np.isnan(X_all)] = 0

    cols = list(Y.columns.values)
    symptoms = ['happy', 'pms', 'sad', 'sensitive_emotion', 'energized', 'exhausted',
                'high_energy', 'low_energy', 'cramps', 'headache', 'ovulation_pain',
                'tender_breasts', 'acne_skin', 'good_skin', 'oily_skin', 'dry_skin']
    with open("result.txt", 'w') as f:
        f.write("user_id,day_in_cycle,symptom,probability\n")
    
    for symptom in symptoms:
        print(symptom)
        s_Y = Y[[x for x in cols if x[1] == symptom]]
        print("Lasso")
        pipeline = Pipeline([
            ('remove_low_variance_features', VarianceThreshold(threshold=0.0)),
            ('standard_scale', StandardScaler()),
            ('estimator', Lasso()),
        ])

        pipeline.fit(X, s_Y.values)

        print("dumping...")
        data_dir = 'data'
        cycles0 = pd.read_csv(join(data_dir, 'cycles0.csv'))
        c_length = {k:v for k,v in zip(cycles0.user_id.values, cycles0.expected_cycle_length)}
        dump(symptom, pipeline, X_all, c_length, data['users'].user_id)
    
if __name__ == '__main__':
    run()
