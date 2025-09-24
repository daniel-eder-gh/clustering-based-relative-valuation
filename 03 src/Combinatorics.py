# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:46:33 2024

@author: Dani
"""

import math

total_features = 18
total_possibilities = 0

min_feat_set_size = 4
max_feat_set_size = 10

for k in range(min_feat_set_size, max_feat_set_size + 1):
    total_possibilities += math.comb(total_features, k)
    
print(total_possibilities)