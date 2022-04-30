from scipy.io import loadmat

#from https://github.com/Muthu-Jeyanthi/ECoG_Hand_Movement_analysis/blob/main/BrainIO%20Hand%20pose%20data%20analysis.ipynb

# Helper Functions

# Create trials from data

def create_trials(data):
  boundary_frames = []
  for i in range(0 , len(data[61, :])):
    if (data[61 , i]!=0) and ((data[61, i-1] == 0) or (data[61,i+1]==0)):
      boundary_frames.append(i)
  
  Ecog_data = []
  Ecog_label = []
  Fingers_data = []
  for i in range(0 , len(boundary_frames) , 2):

    
      Ecog_data.append(data[1:61 , boundary_frames[i]-1000:boundary_frames[i+1]+1001] [: , 0:4400])
      Fingers_data.append( data[62: , boundary_frames[i]-1000:boundary_frames[i+1]+1001] [: , 0:4400])
      Ecog_label.append(data[61 , boundary_frames[i] ])

  return np.stack(Ecog_data) , Ecog_label , np.stack(Fingers_data)
#np.vstack([data[1:61,:] ,data[62: , :]]



# the datastructure is as follows: 67 * 507025 because
# 67 = n_channels
# n_trials = 90 
# each trial has rock paper scissor 2-3 secs and other 2 s of blank screen ~ total 5 s
# so total data points are = 90 * trial duration *frequency ~ 507025
# total experiment duration is 450 secs or so
data = loadmat(r"C:\Users\Shreya Kapoor\Desktop\ECoG\ECoG_Handpose.mat")['y']

ecog_data , ecog_label , fingers_data = create_trials(data)
print(ecog_data.shape) # 90 trials x 60 channels x 4400
print(ecog_label)
print(fingers_data.shape)
'''
After excluding channels that were notably bad due to
high impedance, we re-referenced the data by the common
average. After that, a notch-filter cascade (recursive 6th-order
Butterworth, bandwidth: 5 Hz) up to the 6th harmonic was
used to remove interference peaks from the spectrum at integer
multiples of the power line frequency'''





