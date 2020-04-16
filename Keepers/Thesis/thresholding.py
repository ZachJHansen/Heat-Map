import numpy as np
import pandas
import h5py


# For every timestamp, classify with DNN

# For every timestamp, classify with threshold logic

# I want a 2D matrix: columns = timestamp, DNN classification, threshold classification

# Save to h5 file or something. Maybe CSV. Want to read it into R.

# Given the path to an hdf5 file and a dataset to read
# Return a numpy array representing dataset in file
def read_hd5_file(path, dataset):
    hf = h5py.File(path, 'r')
    n1 = np.array(hf[dataset][:])
    return(n1)

# The chilled water temp is the 15th reading in each feature vector
# If cw_pipe is >= 55 for the entire window, alarm is true for t4
def chilled_water_thresh(window):
    alarm = True
    for i in range(5):
        index = 14 + 52*i
        if (window[index] < 55):
            alarm = False
    return(alarm)

# For a feature vector representing timestamp t4, returns true if 
# 6 or more temperature sensors read above 90 during t4
def hi_temp_thresh(feature_vector):
    #temps1 = temp_extractor(window[:51])
    #temps2 = temp_extractor(window[52:103]
    #temps3 = temp_extractor(window[104:155]
    #temps4 = temp_extractor(window[156:207])
    #temps5 = temp_extractor(window[208:260])
    temps = temp_extractor(feature_vector)
    count = 0
    for t in temps:
        if (t > 90):
            count += 1
    if (count >= 6):
        return(True)
    else:
        return(False)

# Alarm is not “cleared” until the temps have been below 55F for more than 10 consecutive minutes.
# window3 is t+0 -> t+4, window2 is t-5 -> t-1, window1 is t-10 -> t-6
def cw_reset(window1, window2, window3):
    reset = True
    # Is window3 below 55F?
    for i in range(5):
        index = 14 + 52*i
        if (window3[index] >= 55):
            reset = False
            return(reset)
    # Is window2 below 55F?
    for i in range(5):
        index = 14 + 52*i
        if (window2[index] >= 55):
            reset = False
            return(reset)
    # At this point, 10 minutes of data were all below 55
    # Just need one more minute from window1
    index = 14 + 52*4
    if (window1[index] >= 55):
        reset = False
    return(reset)
    
# Once the alarm is triggered, it is not “cleared” until less than 6 sensors are above 90F for 10 consecutive minutes.
def hi_temp_reset(window1, window2, window3):
    reset = True
    indices = [(0,51), (52,103), (104,155), (156,207), (208,260)]
    # Does window3 have > 6 sensors above 90?
    for index_pair in indices:
        fv = window3[index_pair[0]:index_pair[1]]
        alarm = hi_temp_thresh(fv)
        if (alarm): 
            reset = False
            return(reset)
    # Does window2 have > 6 sensors above 90?
    for index_pair in indices:
        fv = window2[index_pair[0]:index_pair[1]]
        alarm = hi_temp_thresh(fv)
        if (alarm): 
            reset = False
            return(reset)
    # Does the last minute of window1 have > 6 sensors above 90?
    fv = window1[208:260]
    alarm = hi_temp_thresh(fv)
    if (alarm): 
        reset = False
    return(reset)

# Temp sensors (readings like _M, _T) are found at indices 2-9 and 16-27
def temp_extractor(feature_vector):
    temps = feature_vector[2:10]
    for val in feature_vector[16:28]:
        temps.append(val)
    return(temps)


# If ALARM is true for windowed_data[i] that means ALARM is true for the LAST timestamp in windowed_data[i], not the first
# Keep in mind these are SLIDING windows, so window0 is t0-t4, window1 is t1-t5, etc.
# But to get > 10 minutes of data PRIOR to windowX, you need windowX-10 + windowX-5
# WindowX = tX - tX+4
# WindowX-5 = tX-5 - tX-1
# WindowX-10 = tX-10 - tX-6

# State 1: CW_ALARM = F, HT_ALARM = F
# State 2: CW_ALARM = T, HT_ALARM = F
# State 3: CW_ALARM = T, HT_ALARM = T
# State 4: CW_ALARM = F, HT_ALARM = T

#windows = read_hd5_file("OrderedTrainingData.hdf5", "fv")
#windows = np.vstack((windows, read_hd5_file("OrderedTestingData.hdf5", "fv")))
#md = read_hd5_file("OrderedTrainingData.hdf5", "md")
#md = np.vstack((md, read_hd5_file("OrderedTestingData.hdf5", "md")))
#md = np.concatenate((md, np.zeros((len(md),1))), axis=1)   # Add a column of all zeroes



# Expects a feature vector matrix (windows), and a metadata matrix with [Predicted Label, Timestamp] columns
# Event number is what event we are on
def thresh(windows, md, event):
    print("Begin Inner Thresh")
    md = np.concatenate((md, np.zeros((len(md),1))), axis=1)   # Add a column of all zeroes
    CW_ALARM = False
    HT_ALARM = False
    for i in range(10, len(windows)):
        win1 = windows[i].tolist()
        win2 = windows[i-5].tolist()
        win3 = windows[i-10].tolist()
        if (not CW_ALARM and not HT_ALARM):     # State 1
            CW_ALARM = chilled_water_thresh(win1)
            HT_ALARM = hi_temp_thresh(win1)
        elif (CW_ALARM and not HT_ALARM):       # State 2
            HT_ALARM = hi_temp_thresh(win1)
            CW_ALARM = not cw_reset(win3, win2, win1)
        elif (CW_ALARM and HT_ALARM):         # State 3
            CW_ALARM = not cw_reset(win3, win2, windows[i])
            HT_ALARM = not hi_temp_reset(win3, win2, win1)
        elif (not CW_ALARM and HT_ALARM):         # State 4
            CW_ALARM = chilled_water_thresh(win1)
            HT_ALARM = not hi_temp_reset(win3, win2, win1)
        else: print("You screwed up your FSM")

        if (CW_ALARM or HT_ALARM):    # Some sort of alarm is present
            md[i,2] = 1
    print("End Inner Thresh")
    pandas.DataFrame(md).to_csv("Event" + str(event) + "-Comparisonss.csv", header=["PredictedLabel", "Timestamp", "ThresholdLabel"])
    







