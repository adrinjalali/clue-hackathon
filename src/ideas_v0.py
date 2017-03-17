# -*- coding: utf-8 -*-
"""
Created on Tue Mar  15 23:23:24 2017

@author: Flaminia
"""

import pandas as pd
import random
import numpy as np
import seaborn as sb

sample = pd.read_csv("data/data_sample.csv") # the first sample, one single file

# List users
udser_id = list(pd.unique(sample.user_id))
# density is a kind of pre processing data
density = pd.DataFrame()

# For each user
for u_id in udser_id:
    single_user = sample[sample.user_id == u_id]
    # Day of ovulation (assuming that the luteal phase lasts 14 days)
    # luteal lenght uniformly distributed around 14 days ±2
    # ov_day becomes the day 0 for each cycle of each user
    ov_day = single_user.cycle_length - (14 + random.randrange(-2, 2, 1))
    #ov_day_mean = int(np.mean(ov_day))
    #cycle_mean = int(np.mean(single_user.cycle_length))  
    #cycle_std = int(np.std(single_user.cycle_length))    

# select "energy" in id1
    user_energy = single_user[single_user.category == 'energy']

    symptom_keys = list(set(user_energy.symptom))
    sk = pd.DataFrame(symptom_keys)
    days = list(set(single_user.day_in_cycle))

    colors = {'exhausted': 'black', 'low_energy': 'blue', 'energized': 'red', 'high_energy': 'purple'}
    color = sk[0].apply(lambda x: colors[x])  # links each symptom with the relative color in colors
    s = 80
    fig1, ax1 = sb.plt.subplots(figsize=(12, 6))

# for each user predictions are made with respect to the day of ovulation, which becomes the day 0
# the day of ovulation represents a fixed point in a cycle length of each user     
    for day in days:
        i = 1
        for key in symptom_keys:
            c = color[i-1]
            points = user_energy.loc[(user_energy['day_in_cycle'] == day) &
                                     (user_energy['symptom'] == key)]
            diary = pd.DataFrame({'user_id': u_id, 'day_to_from_ov':day-ov_day, 'symptom': key,
                                  'occurrence': points.shape[0]})
            density = density.append(diary)
            # %%
            # print(day,key,len(points))
            ax1.scatter(day-ov_day, points.shape[0], s, c, label=key)
            ax1.scatter(0, 0, color='r', s=100, marker='^', alpha=.4, label='ovulation')
            ax1.set_title('User ID:' + str(u_id))
            props1 = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            textstr1 = 'number of cycles= %.0f' % (len(single_user.cycle_length))
            ax1.text(0.55, 0.95, textstr1, transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props1)
            i += 1
            # handles, labels = ax1.get_legend_handles_labels()
            # ax1.legend([handle for i, handle in enumerate(handles) if i == 0],
            #            [label for i, label in enumerate(labels) if i == 0],
            #            loc="upper left", bbox_to_anchor=[0, 1], ncol=1, shadow=True, title="Legend", fancybox=True)

# plt.legend((lo, ll, l, a, h, hh, ho),
#     ('Low Outlier', 'LoLo', 'Lo', 'Average', 'Hi', 'HiHi', 'High Outlier'),
#     scatterpoints=1,
#     loc='lower left',
#     ncol=3,
#     fontsize=8)
sb.plt.show()
