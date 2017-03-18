import pandas as pd
import numpy as np
import os
import csv
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# preparing countries to continent mapping
reader = csv.reader(open('countries_mapping_c.csv', 'r'))
d_continent = {}
for row in reader:
    d_continent[row[1]] = row[0]

reader = csv.reader(open('countries_mapping_r.csv', 'r'))
d_region = {}
for row in reader:
    d_region[row[0]] = row[6]

# preparing data
df_u = pd.read_csv("users.csv")
df_u = df_u.apply(lambda x: x.fillna(x.median()) if np.issubdtype(x.dtype, np.number) else x, axis=0)
df_u['continent'] = df_u.country.map(d)
df_u = df_u[(pd.notnull(df_u['continent']))] #df_u[(pd.isnull(df_u['continent']))]

# preparing the dataset to cluster
drop_elements = ['Oceania', 'user_id', 'country', 'platform', 'continent']
train = pd.concat([df_u, pd.get_dummies(df_u.continent)], axis=1).drop(drop_elements, axis = 1)

scaler = StandardScaler()
train['birthyear'] = scaler.fit_transform(train['birthyear'].reshape(-1, 1))
train['weight'] = scaler.fit_transform(train['birthyear'].reshape(-1, 1))
train['height'] = scaler.fit_transform(train['birthyear'].reshape(-1, 1))

# weird analysis stuff (just some copy-paste code for reference)

H_clusters = 8
df_u['cluster'] = KMeans(n_clusters = H_clusters).fit_predict(train.values)

Z = linkage(train.values, 'ward')

analysis = pd.concat([df_u[['birthyear', 'weight', 'height']].groupby(df_u.cluster).mean(),
    df_u.continent.groupby(df_u.cluster).agg(lambda x:x.value_counts().index[0]),
    df_u.continent_label.groupby(df_u.cluster).mean()], axis = 1)

df_u.cluster.groupby(df_u.continent).mean()

[df_u[['birthyear', 'weight', 'height']].mean(), 
    df_u.continent.value_counts().index[0]]

continent_labels = list(df_u.continent.unique())
df_u['continent_label'] = df_u.continent.map(lambda x: continent_labels.index(x))
