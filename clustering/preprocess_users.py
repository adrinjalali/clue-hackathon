# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:34:56 2017

@author: mskara
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from clustering.users_features import df_users_adj,df_users
from src.pre_process import load_binary


def getDummies(dfz, col, minCtn = 10):
    '''
    function which create dummy variables 
    for the different categories
    '''    
    df2 = dfz.copy()
    df2['_id'] = 1
    df_aux = df2.groupby(col).aggregate({'_id':'count'}).reset_index() 
    df_aux = df_aux[df_aux._id>=minCtn]
    topColTypes = list(set(df_aux[col].values))
    dfz[col] = dfz.apply(lambda r: r[col] if r[col] in topColTypes else 'OTHER' , axis=1)
    dummies = pd.get_dummies(dfz[col], prefix=col) # +'_')
    
    return dummies, topColTypes


def get_monthly_activity(user_df):
    '''calculate how an user was active on a specific month'''
    total_cycles = user_df['cycle_id'].nunique()
    total_entries = user_df['day_in_cycle'].value_counts().sum()
    monthly_activity = total_entries / total_cycles
    
    return monthly_activity


def last_month(user_df):
    '''send back the last month value of a user'''
    count_cycle = user_df['cycle_id'].value_counts()
    last_mon    = count_cycle.head(1).keys()[0]

    return last_mon


def new_user(user_df):
    '''a new user is defined when last_mon<=2'''    
    last_mon = last_month(user_df)
    new_us = 0
    if(last_mon <=2):
        new_us = 1
        
    return new_us


def menstrual_activity(user_df):
    '''send back how active is a person around menstrual time'''
    total_cycle_days = (user_df['day_in_cycle'] <6).sum()
    cycle_activity = total_cycle_days / len(user_df['day_in_cycle'])

    return cycle_activity


def create_info_users(df_active):
    unique_users = df_active.user_id.unique()

    info_users = []
    #tp do: link users.csv e active_days.csv
    for item in unique_users:
        
        user_df = df_active[df_active['user_id']==item]
        user = {}
    
        user['user_id']             = item
        user['activity']            = get_monthly_activity(user_df)
        user['menstrual_activity']  = menstrual_activity(user_df)
        user['relative_activity']   = user['activity'] / user_df['cycle_length'].mean()
        last = user_df['cycle_id']==last_month(user_df)
        user['last_month_activity'] = get_monthly_activity(user_df[last])
        user['new_user']            = new_user(user_df)
    
        info_users.append(user)
    
    return info_users

train = pd.concat([df_users], axis=1) \
                            .drop([ 'country', 'platform','first_havers','menopause', 'continent', \
                                   'birthyear', 'weight', 'height','age_bracket'], axis = 1)


data = load_binary()
df_active  = data['active_days']
df_cycles  = data['cycles']                   

df_active = df_active.join(df_cycles.set_index(['user_id','cycle_id']), on = ['user_id','cycle_id'], how = 'right', lsuffix='_x')
info_users = create_info_users(df_active)
info_df = pd.DataFrame(info_users)

users_data = pd.merge(train, info_df,
              left_on=['user_id'],
              right_on=['user_id'],
              how='inner')
users_data





'''
import seaborn as sns
corr = info_df.corr()
sns.regplot(info_df['menstru_activity'],info_df['last_month_activity'])
sns.heatmap(corr, 
            xticklabels = corr.columns.values,
            yticklabels = corr.columns.values)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA

array_users = users_data.values 
X     = array_users[:, 1:17] 
X_std = StandardScaler().fit_transform(X)

sklearn_pca = sklearnPCA(n_components = 3)
Y_sklearn = sklearn_pca.fit_transform(X_std)
eigenValues = sklearn_pca.explained_variance_ratio_
loadings = sklearn_pca.components_

mu = np.mean(X, axis=0)

nComp = 2
Xhat = np.dot(sklearn_pca.transform(X)[:,:nComp], sklearn_pca.components_[:nComp,:])
Xhat += mu
Xhat=pd.DataFrame(Xhat)

#X = PCA on previous data
info_df['last_month_activity']
info_df['menstru_activity']
X = Xhat.ix[:, '0':'1']
k = 3 # Define the number of clusters in which we want to partion the data
kmeans = KMeans(n_clusters = k) # Run the algorithm kmeans
kmeans.fit(X);
##sklearn.preprocessing.StandardScaler
centroids = kmeans.cluster_centers_ # Get centroid's coordinates for each cluster
labels = kmeans.labels_ # Get labels assigned to each data
colors = ['r.', 'g.','b.'] # Define two colors for the plot below
plt.figure()
for i in range(len(X)):
    plt.plot(X[i,0], X[i,1], colors[labels[i]], markersize = 30)
plt.scatter(centroids[:,0],centroids[:,1], marker = "x", s = 300, linewidths = 5) # Plot centroids
plt.show()




info_df['labels'] = labels

labels = df_a['day_in_cycle']

merged = df_a.join(info_df.set_index(['user_id']), on = ['user_id'], how = 'right', lsuffix='_x')

labels = merged['labels']

###
df_a['day_in_cycle'][labels==1].plot(kind='density')
plt.figure()
df_a['day_in_cycle'][labels==0].plot(kind='density')
plt.hist(df_a['day_in_cycle'][labels==0]/len(df_a['day_in_cycle'][labels==0]))
plt.figure()
df_a['day_in_cycle'][labels==2].plot(kind = 'density')
plt.hist(df_a['day_in_cycle'][labels==2])
'''