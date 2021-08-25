# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 17:50:29 2021

@author: amill
"""


#%%

bins = []
variances = []
means = []
for hist_val, values in d.items():
    bins.append(hist_val)
    var = np.var(values)
    mean = np.mean(values)
    variances.append(var)
    means.append(mean)
    
print(means)
print(variances)