#%%
from turtle import color
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
sampling_freq = 1200
ch_types = ['ecog'] * 60 

info = mne.create_info(ch_names=col_names, ch_types=ch_types, sfreq=sampling_freq)
# Load the data
data = loadmat(r"C:\Users\Shreya Kapoor\Desktop\ECoG\ECoG_Handpose.mat")['y']
# Index 1-60 means channels 2-61
raw = mne.io.RawArray(data[1:61,:], info) # we dont need the first channel
timings = data[0,:]
delta_t = timings[1] - timings[0]
gc.collect()
# CH62: paradigm info (0...relax, 1...fist movement, 2...peace movement, 3...open hand)
paradigm_info = data[61, :]
finger_movement_onsets = data[62:, :]
assert  finger_movement_onsets.shape[0]== 5
raw.load_data().resample(600)  #nyquist theorem
raw.pick_types(ecog=True)
del data

#%%
# show when the screen starts presenting stimulus
# 0 corresponds to the rest timings
event_id = event_id = dict( fist_movement = 1 , peace_movement = 2 , open_hand = 3 )
colors = ['red', 'green', 'blue']
for i in range(1, 4):
    plt.plot(timings, paradigm_info==i, color=colors[i-1])


fix, ax = plt.subplots(5,1)
ax = ax.ravel()
for i in range(5):
    ax[i].plot(timings, finger_movement_onsets[i])

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
# cut the data according to the trial and class
# in total there are 90 trials, each of around 5s

# %%
