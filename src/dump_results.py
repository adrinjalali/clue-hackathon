from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import numpy as np
from os.path import join
import pandas as pd


class DummyModel:
    def __init__(self, constant=0.01):
        self.constant = constant

    def predict(self, X):
        res = list()
        for x in X:
            p = np.zeros(29) + self.constant
            res.append(p)
        return np.array(res)

    def fit(self, X, Y):
        return self

    def score(self, X, Y):
        y_pred = self.predict(X)
        y_true = Y
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def get_params(self, **args):
        return {'constant': self.constant}

    def set_params(self, constant, **args):
        self.constant = constant
        return self

def dump_cycle(f, user, ps, symptom, cl):
    """ Takes predicted values, dumps appropriate results accordingly.
    :param f: output file
    :param user: user id
    :param ps: predictions for this user
    :param symptom: symptom for which these predictions are for
    :param cl: expected cycle length
    """
    y = ps
    y[np.isnan(y)] = 0
    x = np.array(list(range(len(ps))))
    xx = np.linspace(x.min(), x.max(), cl)
    itp = interp1d(x, y, kind='linear')
    window_size, poly_order = 5, 3
    yy_sg = savgol_filter(itp(xx), window_size, poly_order)
    for i in range(int(cl)):
        lp = np.max([.001, np.min([yy_sg[int(i)], .99])])
        f.write("%s,%d,%s,%g\n" % (user, i, symptom, lp))


def dump(symptom, model, X_all, c_length, users):
    with open("result.txt", "a") as f:
        predictions = model.predict(X_all)
        for u, p in zip(users, predictions):
            dump_cycle(f, u, p, symptom, c_length[u])


if __name__ == "__main__":
    model = DummyModel()
    data_dir = 'data'
    cycles0 = pd.read_csv(join(data_dir, 'cycles0.csv'))
    c_length = {k: v for k, v in zip(cycles0.user_id.values, cycles0.expected_cycle_length)}
    X_all = cycles0.copy()
    dump("fluid", model, X_all, c_length)
