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
# In collab with stomperhomp@gmail.com
col_names = [f'CH_{i}' for i in range(2, 62)]
sampling_freq = 1200
ch_types =['misc'] + ['ecog'] * 60 + ['stim']
col_names.extend(['paradigm_info'])
print(len(col_names))
info = mne.create_info(ch_names= ["time"] + col_names, ch_types=ch_types, sfreq=sampling_freq)
# Load the data
data = loadmat(r"C:\Users\Shreya Kapoor\Desktop\ECoG\ECoG_Handpose.mat")['y']

# the datastructure is as follows: 67 * 507025 because
# 67 = n_channels in total
# n_trials = 90 
# each trial has rock paper scissor 2-3 secs and other 2 s of blank screen ~ total 5 s
# so total data points are = 90 * trial duration *frequency ~ 507025
# total experiment duration is 450 secs or so
# Index 1-60 means channels 2-61
raw = mne.io.RawArray(data[0:62,:], info) # we dont need the first channel
timings = data[0,:]
delta_t = timings[1] - timings[0]
gc.collect()
#events = mne.find_events(raw, 'stim') -- doesn't work gives an error no stim channel found
# CH62: paradigm info (0...relax, 1...fist movement, 2...peace movement, 3...open hand)
paradigm_info = data[61, :] # these correspond to the trial onsets
finger_movement_onsets = data[62:, :]
assert  finger_movement_onsets.shape[0]== 5
raw.load_data().resample(600)  #nyquist theorem
events = mne.find_events(raw, stim_channel="paradigm_info")
print(events)
raw.pick_types(ecog=True)
event_id = dict( fist_movement = 1 , peace_movement = 2 , open_hand = 3 )
epochs = mne.Epochs(raw, events, tmin=-0.75, tmax=0.75, event_id=event_id,
                    preload=True, baseline=None)
del data
gc.collect()
#%%
for key in event_id.keys():
    evoked = epochs[key].average()
    evoked.plot()
#%%
#%%
'''
event_id_rev = {k:v for v,k in event_id.items()}
trial_onsets = timings[paradigm_info!=0]
# since we want the onsets to be in s
#annotations = mne.Annotations(onset, duration, description)
# we need a format of onset, duration and description
duration_trials = np.repeat(2, len(trial_onsets)) # 160 seconds is the trial duration
description = [event_id_rev[id] for id in paradigm_info if id!=0]
annotations = mne.Annotations(trial_onsets, duration_trials, description)
raw.set_annotations(annotations)
events, event_id = mne.events_from_annotations(raw)
'''
#%%
'''

epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=2,
                    baseline=(None, 0),
                    preload=False)
fig = epochs.plot(events=events)'''
gc.collect()
#%%
#%%
# show when the screen starts presenting stimulus
# 0 corresponds to the rest timings
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
del events
gc.collect()
#%%
# They set the trial length to 0.75 seconds pre- and post-onset, respectively
#full_mean = raw._data.mean()
#mean_col =# get the meaning of the column
#raw._data -= full_mean # this could cause bias though
raw.set_eeg_reference('average')
nf = 50
hf = 150
raw.filter(1, hf)
raw.notch_filter([nf], notch_widths=2, trans_bandwidth=1) 
#raw.notch_filter([50], trans_bandwidth=5)

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

#%%
del  raw
#%%
# %%
# Define a monte-carlo cross-validation generator (reduce variance):
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
epochs_train = epochs.copy().crop(tmin=-0.75, tmax=0.75)
labels = epochs.events[:, -1] - 2
scores = []
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()
cv = ShuffleSplit(5, test_size=0.15, random_state=88)
cv_split = cv.split(epochs_data_train)

# Assemble a classifier
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=5, reg=None, log=True, norm_trace=False)

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([('CSP', csp), ('LDA', lda)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                         class_balance))

# plot CSP patterns estimated on full data for visualization
csp.fit_transform(epochs_data, labels)
#%%
#csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
# %%
w = 10
h = 3

plt.figure(figsize=(16,12))

for i,f in enumerate(csp.patterns_[:h*w]):
    plt.axis("off")
    ax = plt.subplot(h, w, i+1)
    ax.set_title(f"{i+1}")
    ax.imshow(f.reshape((6, 10)).T)

plt.show()
# %%
