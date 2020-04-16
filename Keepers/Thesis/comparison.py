import numpy as np
import scipy
import pandas
import sklearn
import h5py
import sys
import time
import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import *

from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler

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


# Expects a feature vector matrix (windows), and a metadata matrix with [Predicted Label, Actual Label, Timestamp, CW_TEMP, CRAC1_INLET, CRAC2_INLET] columns
# Event number is what event we are on
def thresh(windows, md, event):
    alarm_count = 0
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
            md[i-10,3] = 1
            alarm_count += 1
    print("Event", event, "- this event triggered", alarm_count, "minutes of alarms")
    pandas.DataFrame(md).to_csv("Event" + str(event) + "-Comparisons.csv", header=["PredictedLabel", "ActualLabel", "Timestamp", "CW_TEMP", "CRAC1_INLET", "CRAC2_INLET", "ThresholdLabel"])

# Given the path to an hdf5 file and a dataset to read
# Return a numpy array representing dataset in file
def read_hd5_file(path, dataset):
    hf = h5py.File(path, 'r')
    n1 = np.array(hf[dataset][:])
    return(n1)

# Given a window length, generate array of tuples
# Tuples contain start datetime, end datetime, one week seperation
def date_gen_week(N):
    dates = []
    # Initialize first week
    start_date = datetime.datetime(2018, 12, 19, 21, 30, 0, 0)
    end_date = datetime.datetime(2018, 12, 26, 21, 30, 0, 0)
    tupleware = (start_date.strftime("%Y-%m-%d %H:%M:%S"), end_date.strftime("%Y-%m-%d %H:%M:%S"))
    dates.append(tupleware)
    while (True):
        start_date = end_date - datetime.timedelta(minutes=N) + datetime.timedelta(minutes=2)
        end_date = end_date + datetime.timedelta(days=7)
        tupleware = (start_date.strftime("%Y-%m-%d %H:%M:%S"), end_date.strftime("%Y-%m-%d %H:%M:%S"))
        dates.append(tupleware)
        if (end_date >= datetime.datetime(2019, 11, 1, 12, 0, 0, 0)):
            break
    return(dates)

# Stitches data from all files into one Numpy 2D array
def consolidator(mx_type, N):
    # Generate ordered list of file names
    file_names = []
    dates = date_gen_week(N)
    for d in dates:
        d1 = d[0].split(" ")[0]
        d2 = d[1].split(" ")[0]
        file_names.append(d1 + "-" + d2 + ".hdf5")

    # Create ordered dataset from the filename list
    init = read_hd5_file("../Data/"+file_names[0], mx_type)
    for i in range(1, len(file_names)):
        try:
            data = read_hd5_file("../Data/"+file_names[i], mx_type)
            init = np.vstack((init, data))
        except Exception as e:
            print("Error while consolidating data from file", file_names[i], "Exception:", e, file=open("TrainingLog.txt", "a"))
    return(init)

def mostly(fv, md, block, i):
    total = 0
    start = block[0]
    end = block[1]
    lbls = np.array(md[start:end,0], dtype='int')
    try:
        event = True
        neg, pos = np.bincount(lbls)
        print("Event: ", start, "-", end)
        print("Positive:", pos, "| Negative:", neg)
    except:
        event = False
        print("Not Event: ", start, "-", end)
    if event:
        try:
            # Scale Data
            #fv_scaley = scaler.fit_transform(fv[start:end,])

            # Reshape data
            #fv_shapely = np.reshape(fv_scaley, (len(fv_scaley), 5, 52))
            #print("FV Reshaped:", fv_shapely.shape)

            # Predict labels
            #preds = np.round(model.predict(fv_shapely, batch_size=16), 2)
            preds = np.zeros((len(md[start:end,]), 1))
            # Format: 3 columns, predicted value, actual label, timestamp 
            md_new = np.concatenate((preds, np.reshape(md[start:end,0:2], (len(md[start:end]),2))), axis=1)
            temp = fv[start:end]

            # Just take the value corresponding to timestamp 1 for now, change to last timestamp if needed
            cw = np.reshape(temp[:, 14], (len(temp), 1))
            c1 = np.reshape(temp[:, 10], (len(temp), 1))
            c2 = np.reshape(temp[:, 12], (len(temp), 1))
            md_new = np.concatenate((md_new, cw, c1, c2), axis=1)
            print("New MD Shape:", md_new.shape)
            print(md_new[0,:])
            print("Begin Thresh")
            thresh(fv[start-10:end,], md_new, i)                   # The thresholding needs timestamps from 10 minutes prior to each timestamp
            print("End Thresh")
            i += 1
        except Exception as e:
            print("Error:", e)
    return(i)


# Main
testing_blocks = [
(409461, 409615),
(363382, 363591),
(0, 28257),
(152264, 162264),
(29681, 39681),
(302264, 318928),
(409615, 419615)]

validation_blocks = [
(318928, 319131),
(29546, 29681),
(319131, 324662),
(79681, 89681),
(252264, 262264),
(429615, 444091),
(49681, 59681)]

training_blocks = [
(343551, 343862),
(358415, 359911),
(419615, 429615),
(262264, 272264),
(142264, 152264),
(212264, 222264),
(172264, 182264),
(69681, 79681),
(282264, 292264),
(383591, 401120),
(324799, 341585),
(242264, 252264),
(373591, 383591),
(182264, 192264),
(59681, 69681),
(344112, 344297),
(343324, 343360),
(202264, 212264),
(119681, 132120),
(292264, 302264),
(89681, 99681),
(346193, 354630),
(28400, 29546),
(162264, 172264),
(272264, 282264),
(192264, 202264),
(109681, 119681),
(232264, 242264),
(444252, 463247),
(361560, 363382),
(99681, 109681),
(132264, 142264),
(39681, 49681),
(222264, 232264),
(363591, 373591),
(401396, 409461),
(132120, 132264),
(401120, 401396),
(28257, 28400),
(444091, 444252),
(324662, 324799)]

# Read in testing data
fv = consolidator("fv", 5)
md = consolidator("md", 5)

print("FV Shape:", fv.shape)
print("MD Shape:", md.shape)

scaler = StandardScaler()

# Load model
#model = load_model("../Furtherer/LSTM/Models/m.h5")

i = 0
print("\nBeginning the Testing Blocks!")
for block in testing_blocks:
    i = mostly(fv, md, block, i)
print("\nBeginning the Validation Blocks!")
for block in validation_blocks:
    i = mostly(fv, md, block, i)
print("\nBeginning the Training Blocks!")
for block in training_blocks:
    i = mostly(fv, md, block, i)

if False:
    md = consolidator("md", 5)
    print(md[:5,:])
    #results = pandas.read_csv("Event0-Comparisonss.csv", names=["PredictedLabel", "ActualLabel", "Timestamp", "ThresholdLabel"])
    #lbls = results['ThresholdLabel'].value_counts()
    #print(lbls)








