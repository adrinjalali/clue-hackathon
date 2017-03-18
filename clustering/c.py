import pandas as pd
import numpy as np
import os
import csv

reader = csv.reader(open('countries_mapping.csv', 'r'))
d = {}
for row in reader:
    d[row[1]] = row[0]

df_u = pd.read_csv("users.csv")
df_u = df_u.apply(lambda x: x.fillna(x.median()) if np.issubdtype(x.dtype, np.number) else x, axis=0)
df_u['continent'] = df_u.country.map(d)
df_u = df_u[(pd.notnull(df_u['continent']))]

drop_elements = ['Oceania', 'user_id', 'country', 'platform', 'continent']
train = pd.concat([df_u, pd.get_dummies(df_u.continent)], axis=1).drop(drop_elements, axis = 1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train['birthyear'] = scaler.fit_transform(train['birthyear'].reshape(-1, 1))
train['weight'] = scaler.fit_transform(train['birthyear'].reshape(-1, 1))
train['height'] = scaler.fit_transform(train['birthyear'].reshape(-1, 1))

from sklearn.cluster import KMeans
H_clusters = 8
df_u['cluster'] = KMeans(n_clusters = H_clusters).fit_predict(train.values)