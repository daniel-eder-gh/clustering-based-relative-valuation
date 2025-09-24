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


path = r'C:\Users\Dani\OneDrive - Wirtschaftsuniversität Wien - IT-SERVICES\Dokumente\WU\Bachelorarbeit\02 Code\07 Performance Evaluation'
with open(f'{path}\DBSCAN_minpts_Growth_valuation_output_robust_all_quarters.pkl', 'rb') as f:
    valuation_df_backup = pickle.load(f)
    
data_quarters_df, strat_data, rets_quick_result = val_mod.trading_strategy(valuation_df_backup)

print("Mean returns at a glance:", rets_quick_result)

rets = val_mod.return_calc(data_quarters_df)
#rets.to_excel(f'{path}\DBSCAN_minpts_{g_or_v}_returns_robust_all_quarters.xlsx')
rets.iloc[:,-3:].plot()

for r in rets.columns[1:4]:
    print(f"Return stats for {r}:")
    print("Annual return: ",val_mod.annualize_rets(rets[r], 4))
    print("Volatility: ",val_mod.annualize_vol(rets[r], 4))
    print("Sharpe ratio: ",val_mod.sr_calc(rets[r]))
    
    