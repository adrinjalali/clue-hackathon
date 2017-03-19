# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 15:06:05 2017

@author: mskara
"""

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

# preparing countries to continent mapping
reader = csv.reader(open('countries_mapping.csv', 'r'))
d = {}
for row in reader:
    d[row[1]] = row[0]

# preparing data
df_users = pd.read_csv("users.csv") \
    .apply(lambda x: x.fillna(x.median()) if np.issubdtype(x.dtype, np.number) else x, axis=0)
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

# preparing the dataset to cluster
df_users_adj = pd.concat([df_users, pd.get_dummies(df_users.continent), pd.get_dummies(df_users.age_bracket)], axis=1) \
                            .drop(['Oceania', 'country', 'platform', 'continent', \
                                   'birthyear', 'weight', 'height','age','age_bracket'], axis = 1)


