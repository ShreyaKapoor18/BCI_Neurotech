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
from mne.time_frequency import fit_iir_model_raw
import gc
from scipy import signal
gc.enable()

col_names = [f'CH_{i}' for i in range(2, 62)]
col_names.extend(['paradigm_info'])
print(len(col_names))
sampling_freq = 1200
ch_types = ['ecog'] * 60 + ['misc']

info = mne.create_info(ch_names=col_names, ch_types=ch_types, sfreq=sampling_freq)
# Load the data
data = loadmat(r"C:\Users\Shreya Kapoor\Desktop\ECoG\ECoG_Handpose.mat")['y']
raw = mne.io.RawArray(data[:61,:], info)
gc.collect()

raw.load_data().resample(600)  #nyquist theorem
raw.pick_types(ecog=True)
#%%
timings = data[0,:]
delta_t = timings[1] - timings[0]
del data
#%%
'''
After excluding channels that were notably bad due to
high impedance, we re-referenced the data by the common
average. After that, a notch-filter cascade (recursive 6th-order
Butterworth, bandwidth: 5 Hz) up to the 6th harmonic was
used to remove interference peaks from the spectrum at integer
multiples of the power line frequency'''

# They set the trial length to 0.75 seconds pre- and post-onset, respectively
full_mean = raw._data.mean()
#mean_col =# get the meaning of the column
raw._data -= full_mean # this could cause bias though
raw.notch_filter([60], trans_bandwidth=5)

#picks = mne.pick_types(raw.info, meg='grad', exclude='bads')

order = 5  # define model order
#picks = picks[:1]

# from https://mne.tools/stable/auto_examples//time_frequency/temporal_whitening.html#sphx-glr-auto-examples-time-frequency-temporal-whitening-py
# Estimate AR models on raw data
b, a = fit_iir_model_raw(raw, order=order, picks=['ecog'], tmin=timings[0], tmax=timings[-1])
d, times = raw[0, 10000:20000]  # look at one channel from now on
d = d.ravel()  # make flat vector
innovation = signal.convolve(d, a, 'valid')
d_ = signal.lfilter(b, a, innovation)  # regenerate the signal
d_ = np.r_[d_[0] * np.ones(order), d_]  # dummy samples to keep signal length

plt.close('all')
plt.figure()
plt.plot(d[:100], label='signal')
plt.plot(d_[:100], label='regenerated signal')
plt.legend()

plt.figure()
plt.psd(d, Fs=raw.info['sfreq'], NFFT=2048)
plt.psd(innovation, Fs=raw.info['sfreq'], NFFT=2048)
plt.psd(d_, Fs=raw.info['sfreq'], NFFT=2048, linestyle='--')
plt.legend(('Signal', 'Innovation', 'Regenerated signal'))
plt.show()
#%%
#raw.plot()
#plt.savefig('results/mne_plot.png')
#picks=['CH_5', 'CH_10', 'CH_15']

#raw.plot_psd(average=True)
#plt.savefig('results/plot_spectral_density.png')

#plt.savefig('results/ecog.png')
#raw.plot_sensors(ch_type='ecog')

# Then we remove line frequency interference



#%%

events, event_id = mne.events_from_annotations(raw)
print(event_id, events)
# %%
