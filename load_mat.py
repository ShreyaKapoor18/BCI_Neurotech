#%%
import pandas as pd
import numpy as np
from scipy.io import loadmat
#%%
data = loadmat(r"C:\Users\Shreya Kapoor\Desktop\ECoG\ECoG_Handpose.mat")
# %%
'''
CH1: sample time
CH2-61: ECoG (raw and DC-coupled; recorded from right sensorimotor cortex)
CH62: paradigm info (0...relax, 1...fist movement, 2...peace movement, 3...open hand)
CH63: data glove thumb
CH64: data glove index
CH65: data glove middle
CH66: data glove ring
CH67: data glove little'''

# first row corresponds to the timings
data_only = pd.DataFrame(data['y'])
del data
#%%
data_only = data_only.transpose()
#%%
col_names = [f'CH_{i}' for i in range(1, 62)]
col_names.extend(['paradigm_info','data_glove_thumb', 'data_glove_index','data_glove_middle', 'data_glove_ring', 'data_glove_little' ])
data_only.columns = col_names
#%%
# now the shape will be 507025 x 67
data_only.iloc[:, 61:].describe()
#%%
# Time series plotting
# %%
X = data_only.iloc[:, :62]
y = data_only['paradigm_info']
print(X.columns)
del data_only
#%%
import matplotlib.pyplot as plt
#%%
# for the non rest period
#n_r = X[X.CH_1!=0]
fig, ax = plt.subplots(60,1, figsize=(20,50))
ax= ax.ravel()
for i in range(1,61): #since the first channel is for the timings, ignore index 0
    ax[i-1].plot(X.iloc[::1200, 0], X.iloc[::1200, i])
    ax[i-1].set_xlabel('time')
    ax[i-1].set_ylabel(col_names[i])
    
plt.savefig('Initial_view.jpg')
#%%