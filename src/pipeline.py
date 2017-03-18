import numpy as np
import sys
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import pandas as pd

from src.pre_process import process_level2, load_binary


def run():
    data = load_binary()

    # (Opt: user clustering)

    # Extract features
    user_feat_matrix = process_level2(data)  # X

    X = user_feat_matrix['X']
    X[np.isnan(X)] = 0
    Y = user_feat_matrix['Y']
    Y[np.isnan(Y)] = 0
    X_all = user_feat_matrix['X_all']
    X_all[np.isnan(X_all)] = 0

    cols = list(Y.columns.values)
    symptoms = ['happy', 'pms', 'sad', 'sensitive_emotion', 'energized', 'exhausted',
                'high_energy', 'low_energy', 'cramps', 'headache', 'ovulation_pain',
                'tender_breasts', 'acne_skin', 'good_skin', 'oily_skin', 'dry_skin']
    user_ids = X[('user_id', '', '', '')].copy()
    del X[('user_id', '', '', '')]
    for symptom in symptoms:
        s_Y = Y[[x for x in cols if x[3] == symptom]]
        pipeline = Pipeline([
            ('remove_low_variance_features', VarianceThreshold(threshold=0.0)),
            ('standard_scale', StandardScaler()),
            ('estimator', LinearRegression()),
        ])

        pipeline.fit(X.values, s_Y.values)
    # (Opt: Select features)
    pass

    # Pre-predict processing
    pass

    # Predict
    pass


    # Post-predict processing
    pass



    # save results in the correct format.
    results.to_csv('./result.txt', index=None)


if __name__ == '__main__':
    run()
