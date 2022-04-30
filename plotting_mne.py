#%%
import mne
import os.path as op

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
#from mne_bids import BIDSPath, read_raw_bids
from scipy.io import loadmat
import mne
from mne.viz import plot_alignment, snapshot_brain_montage
import gc
gc.enable()
#%%
col_names = [f'CH_{i}' for i in range(2, 62)]
col_names.extend(['paradigm_info'])
print(len(col_names))
sampling_freq = 1200
ch_types = ['ecog'] * 60 + ['misc']
info = mne.create_info(ch_names=col_names, ch_types=ch_types, sfreq=sampling_freq)
data = loadmat(r"C:\Users\Shreya Kapoor\Desktop\ECoG\ECoG_Handpose.mat")
raw = mne.io.RawArray(data['y'][1:62,:], info)
del data
raw.pick_types(ecog=True)

# Load the data
raw.load_data().resample(600)

#raw.plot()
#plt.savefig('results/mne_plot.png')
picks=['CH_5', 'CH_10', 'CH_15']

raw.plot_psd(average=True)
#plt.savefig('results/plot_spectral_density.png')

plt.savefig('results/ecog.png')
#raw.plot_sensors(ch_type='ecog')

# Then we remove line frequency interference
raw.notch_filter([60], trans_bandwidth=3)

#%%

events, event_id = mne.events_from_annotations(raw)
print(event_id, events)
# %%
