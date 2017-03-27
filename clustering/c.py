from __future__ import division
import pandas as pd
import numpy as np
import os
import csv
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import itertools
from pandas.tools.plotting import scatter_matrix

# preparing countries to continent mapping
reader = csv.reader(open('countries_mapping.csv', 'r'))
d = {}
for row in reader:
    d[row[1]] = row[0]

# preparing data
df_u = pd.read_csv("users.csv") \
    .apply(lambda x: x.fillna(x.median()) if np.issubdtype(x.dtype, np.number) else x, axis=0)
df_u['continent'] = df_u.country.map(d).fillna("Oceania")
df_u.continent = df_u.continent.apply(lambda x: 'Asia' if x == 'South Korea' else x)
# df_u['phase'] = ((df_u.birthyear - df_u.birthyear.min())/(df_u.birthyear.max() - df_u.birthyear.min()))
# df_u['phase'] = df_u.phase.apply(lambda x: 0 if x > 0.8 else 2 if x < 0.2 else 1)
# df_u['adolescence'] = (((df_u.birthyear - df_u.birthyear.min())/(df_u.birthyear.max() - df_u.birthyear.min())) > 0.8) * 1
# df_u['menopause'] = (((df_u.birthyear - df_u.birthyear.min())/(df_u.birthyear.max() - df_u.birthyear.min())) < 0.2) * 1
df_u['age'] = 2017 - df_u.birthyear
df_u['age_bracket'] = df_u.age.apply(lambda x: 0 if x < 18 else 2 if x > 24 else 1)
df_u['first_havers'] = (2017 - df_u.birthyear).apply(lambda x: 1 if x < 15 else 0)
df_u['menopause'] = (((df_u.birthyear - df_u.birthyear.min())/(df_u.birthyear.max() - df_u.birthyear.min())) < 0.2) * 1
df_u['bmi'] = df_u.weight / ((df_u.height/100)**2)

# preparing the dataset to cluster
train = pd.concat([df_u, pd.get_dummies(df_u.continent)], axis=1) \
    .drop(['Oceania', 'country', 'platform', 'continent', 'user_id',\
    'birthyear', 'weight', 'height'], axis = 1)

scaler = StandardScaler()
train['age'] = scaler.fit_transform(train['age'].reshape(-1, 1))
train['bmi'] = scaler.fit_transform(train['bmi'].reshape(-1, 1))

scatter_matrix(train.drop(['age', 'first_havers', 'menopause'], axis = 1), diagonal='kde')
plt.show(block = False)
train.drop(['age', 'first_havers', 'menopause'], axis = 1).corr()

# weird analysis stuff (just some copy-paste code for reference)

# H_clusters = 8
# df_u['cluster'] = KMeans(n_clusters = H_clusters).fit_predict(train.values)

distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters = i, init='k-means++', n_init=10, max_iter=300)
    km.fit(train.values)
    distortions.append(km.inertia_)

plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show(block = False)

analysis = pd.concat([df_u[['birthyear', 'weight', 'height']].groupby(df_u.cluster).mean(),
    df_u.continent.groupby(df_u.cluster).agg(lambda x:x.value_counts().index[0])], axis = 1)

# df_u.cluster.groupby(df_u.continent).mean()

# [df_u[['birthyear', 'weight', 'height']].mean(), 
#     df_u.continent.value_counts().index[0]]

# continent_labels = list(df_u.continent.unique())
# df_u['continent_label'] = df_u.continent.map(lambda x: continent_labels.index(x))

# fig = scatterplot_matrix(train.values, list(train.columns),
#             linestyle='none', marker='o', color='black', mfc='none')
# fig.suptitle('Simple Scatterplot Matrix')
# plt.show()

# def scatterplot_matrix(data, names, **kwargs):
#     """Plots a scatterplot matrix of subplots.  Each row of "data" is plotted
#     against other rows, resulting in a nrows by nrows grid of subplots with the
#     diagonal subplots labeled with "names".  Additional keyword arguments are
#     passed on to matplotlib's "plot" command. Returns the matplotlib figure
#     object containg the subplot grid."""

#     numvars, numdata = data.shape
#     fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8,8))
#     fig.subplots_adjust(hspace=0.05, wspace=0.05)

#     for ax in axes.flat:
#         # Hide all ticks and labels
#         ax.xaxis.set_visible(False)
#         ax.yaxis.set_visible(False)

#         # Set up ticks only on one side for the "edge" subplots...
#         if ax.is_first_col():
#             ax.yaxis.set_ticks_position('left')
#         if ax.is_last_col():
#             ax.yaxis.set_ticks_position('right')
#         if ax.is_first_row():
#             ax.xaxis.set_ticks_position('top')
#         if ax.is_last_row():
#             ax.xaxis.set_ticks_position('bottom')

#     # Plot the data.
#     for i, j in zip(*np.triu_indices_from(axes, k=1)):
#         for x, y in [(i,j), (j,i)]:
#             axes[x,y].plot(data[x], data[y], **kwargs)

#     # Label the diagonal subplots...
#     for i, label in enumerate(names):
#         axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
#                 ha='center', va='center')

#     # Turn on the proper x or y axes ticks.
#     for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
#         axes[j,i].xaxis.set_visible(True)
#         axes[i,j].yaxis.set_visible(True)

#     return fig