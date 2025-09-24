# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:07:02 2024

@author: Dani
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import random
import pickle
import json
import warnings
import time
warnings.filterwarnings('ignore')

import sys
import clustering_v015_final as clust_mod
import valuation_trading_strategy_v04_final as val_mod

g_or_v = "Growth"

### STORAGE - CLUSTERS

path = r'C:\Users\Dani\OneDrive - Wirtschaftsuniversität Wien - IT-SERVICES\Dokumente\WU\Bachelorarbeit\02 Code\07 Performance Evaluation'  
with open(f'{path}\DBSCAN_minpts_{g_or_v}_value_features_cluster_output_robust_all_quarters.pkl', 'rb') as f:
    primary_industries_clustered_backup = pickle.load(f)

### TRADING STRATEGY
# Group the Clusters to prep for output:
grouped_dfs,peer_groups_every_quarter = clust_mod.group_output(primary_industries_clustered_backup)

#run multiple calculation, forming of valuation dataframe, 
#generating trading strategy signals

#but before that, the data in its original form (unscaled) has to be retrieved

raw_data = clust_mod.input(g_or_v,
    r"C:\Users\Dani\OneDrive - Wirtschaftsuniversität Wien - IT-SERVICES\Dokumente\WU\Bachelorarbeit\02 Code\03 Input Data\clustering_data_unscaled.csv")

#form valuation_df
valuation_df = val_mod.get_valuation_df(raw_data, peer_groups_every_quarter)

### STORAGE - VALUATION DF
with open(f'{path}\DBSCAN_minpts_{g_or_v}_value_features_valuation_output_robust_all_quarters.pkl', 'wb') as f:
    pickle.dump(valuation_df, f)
    
'''
with open('valuation_results_dbscan_primary_industry_unscaled.pkl', 'rb') as f:
    valuation_df_backup = pickle.load(f)
'''

data_quarters_df, strat_data, rets_quick_result = val_mod.trading_strategy(valuation_df)

print("Mean returns at a glance:", rets_quick_result)

rets = val_mod.return_calc(data_quarters_df)
rets.to_excel(f'{path}\DBSCAN_minpts_{g_or_v}_value_features_returns_robust_all_quarters.xlsx')
rets.iloc[:,-3:].plot()

for r in rets.columns[1:4]:
    print(f"Return stats for {r}:")
    print("Annual return: ",val_mod.annualize_rets(rets[r], 4))
    print("Volatility: ",val_mod.annualize_vol(rets[r], 4))
    print("Sharpe ratio: ",val_mod.sr_calc(rets[r]))
    