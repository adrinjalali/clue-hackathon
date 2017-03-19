from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import sys
import numpy as np
from os.path import join
import os
import pandas as pd

class dummy_model:
    def predict(self, X):
        res = list()
        for x in X:
            p = np.zeros(29)
            p[5] = 1
            p[6] = 1
            res.append(p)
        return res

def dump_cycle(f, user, ps, symptom, cl):
    l = len(ps)
    y = ps
    y[np.isnan(y)] = 0
    x = np.array(list(range(len(ps))))
    xx = np.linspace(x.min(),x.max(), cl)
    itp = interp1d(x,y, kind='linear')
    window_size, poly_order = 5, 3
    yy_sg = savgol_filter(itp(xx), window_size, poly_order)
    for i in range(int(cl)):
        lp = np.max([.001,np.min([yy_sg[int(i)], .999])])
        f.write("%s,%s,%d,%g\n"%(user, symptom, i, lp))

def dump(symptom, model, X_all, c_length, users):
    with open("result.txt", "a") as f:
        predictions = model.predict(X_all)
        for u, p in zip(users, predictions):
            dump_cycle(f, u, p, symptom, c_length[u])
        

if __name__ == "__main__":
    model = dummy_model()
    data_dir = 'data'
    cycles0 = pd.read_csv(join(data_dir, 'cycles0.csv'))
    c_length = {k:v for k,v in zip(cycles0.user_id.values, cycles0.expected_cycle_length)}
    X_all = cycles0.copy()
    dump("fluid", model, X_all, c_length)
