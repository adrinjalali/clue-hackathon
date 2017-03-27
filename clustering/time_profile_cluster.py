# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:21:47 2017

@author: mskara
"""
import pandas as pd
import matplotlib.pyplot as plt
from src.pre_process import load_binary

def create_profile_for_symptoms(df, date_range=15):
    profiles = {}
    for symptom in symptoms:
        temp = df[df['symptom'] == symptom]
        sympt_profile = temp.groupby(by=temp['day_in_cycle']).mean()[0:date_range]
        plt.plot(sympt_profile)
        profiles[symptom] = sympt_profile

    return profiles


def check_probability_access(data):
    '''find probability_access'''

    df_active  = data['active_days']
    df_cycles  = data['cycles']
    access_prob = []
    for i in range(1, 30):
        access_prob.append((df_active['day_in_cycle'] == i).sum()
                          /(df_cycles['cycle_length'][df_cycles['cycle_length']>=i]).count())
        
        # access_prob.plot(X)

    return access_prob

df = pd.read_csv('result.txt')
# now is done until 15 day, afterwords our predictions are wrong
daily_profiles = create_profile_for_symptoms(df,date_range = 15)

data = load_binary()
access_profile = check_probability_access(data)
plt.plot (access_profile[0:29])  # probability of access


for symptom in symptoms:
    real_prob = daily_profiles[symptom].copy()
    for i in range(15):
        real_prob.loc[i]=real_prob.loc[i]/access_profile[i]

    plt.plot(real_prob)


