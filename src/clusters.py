# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 15:45:16 2017

@author: mskara
"""
##labelling

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from clustering.preprocess_users import users_data


def cluster_model(users_data, num_cluster=3):
    
    array_users = users_data.values 
    X     = array_users[:, 1:17] 
    X_std = StandardScaler().fit_transform(X)

    sklearn_pca = sklearnPCA(n_components = 3)
    Y_sklearn = sklearn_pca.fit_transform(X_std)
    eigenValues = sklearn_pca.explained_variance_ratio_
    loadings = sklearn_pca.components_

    mu = np.mean(X, axis=0)

    nComp = 2
    Xhat  = np.dot(sklearn_pca.transform(X)[:,:nComp], sklearn_pca.components_[:nComp,:])
    Xhat  = mu + Xhat
    Xhat  = pd.DataFrame(Xhat)

    # X = PCA on previous data
    X = Xhat.ix[:, '0':'1']
    k = num_cluster  # Define the number of clusters in which we want to partion the data
    kmeans = KMeans(n_clusters = k)  # Run the algorithm kmeans
    kmeans.fit(X);
    ##sklearn.preprocessing.StandardScaler
    centroids = kmeans.cluster_centers_  # Get centroid's coordinates for each cluster
    labels = kmeans.labels_  # Get labels assigned to each data
    final_labels = users_data[['user_id']]
    final_labels['labels'] = pd.DataFrame(labels)

    return final_labels

final_labels = cluster_model(users_data, num_cluster=3)
# labels1 = df_active['day_in_cycle']
# merged = df_active.join(info_df.set_index(['user_id']), on = ['user_id'], how = 'right', lsuffix='_x')
# labels2 = merged['labels']


'''
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
'''

