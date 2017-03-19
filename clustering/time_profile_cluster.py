# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:21:47 2017

@author: mskara
"""
import pandas as pd
import matplotlib.pyplot as plt

df_active  = pd.read_csv("active_days.csv")
df_cycles  = pd.read_csv("cycles.csv")
df_users0  = pd.read_csv("users0.csv")


def check_probability_access(df_active, df_cycles):
    '''find probability_access'''
    access_prob = []
    for i in range(1,30) :
        access_prob.append((df_active['day_in_cycle'] == i).sum()/(df_cycles['cycle_length']).count())
        access_prob.plot(X)

    return access_prob

check_probability_access(df_active, df_cycles)

