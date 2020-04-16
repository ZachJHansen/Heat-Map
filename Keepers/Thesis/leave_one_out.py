import scipy
import numpy as np
import matplotlib
import pandas
import sklearn
import statsmodels
import h5py
import sys
import datetime
import time
import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, ELU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
import random
from model_testing import *

def create_model(layer_width, in_dim, dropout):
    METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc')]

    SEED = 4

    # Define model
    model = Sequential()
    model.add(Dense(layer_width, input_dim = in_dim, activation='relu'))
    model.add(Dropout(dropout, noise_shape=None, seed=SEED))
    model.add(Dense(layer_width, activation='relu'))
    model.add(Dropout(dropout, noise_shape=None, seed=SEED))
    model.add(Dense(layer_width, activation='relu'))
    model.add(Dropout(dropout, noise_shape=None, seed=SEED))
    model.add(Dense(layer_width, activation='relu'))
    model.add(Dropout(dropout, noise_shape=None, seed=SEED))
    model.add(Dense(layer_width, activation='relu'))
    model.add(Dropout(dropout, noise_shape=None, seed=SEED))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)
    return(model)

def train_model(model, fv_train, md_train, epochs, batch, event_weight, run):
    # Scale data
    scaler = StandardScaler()
    fv_train = scaler.fit_transform(fv_train)
    The_Weight = {0: 1 - event_weight, 1: event_weight}
    
    # Fit the model
    try:
        hist = model.fit(x=fv_train, y=md_train, batch_size=batch, epochs=epochs, verbose=0, class_weight=The_Weight, shuffle=False)

        # Save the model
        model.save("Model" + run + ".h5")

        # Record training history in a JSON file
        df = pandas.DataFrame(hist.history)
        hist_json = "Run" + run + "-history.json"
        with open(hist_json, "w") as f:
            df.to_json(f)
            f.close()
        return(model)
    except Exception as e:
        print("Something's gone wrong during model training:", e)
        return(None)

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
    init = read_hd5_file("/home/zhansen/DataPrep/Windows/NewDeal/Data/"+file_names[0], mx_type)
    for i in range(1, len(file_names)):
        try:
            data = read_hd5_file("/home/zhansen/DataPrep/Windows/NewDeal/Data/"+file_names[i], mx_type)
            init = np.vstack((init, data))
        except Exception as e:
            print("Error while consolidating data from file")
    return(init)

# Print the distribution of all tuples (positive/negative labels)
# Return a list of index tuples [(start, end), ...]
def get_event_splits():
    # Read in all data
    fv = consolidator("fv", 5)
    md = consolidator("md", 5)
    in_dim = len(fv[0])

    # What indices should I split the data at?
    indices = []
    EVENT = False
    for i in range(len(md)-60):
        window = md[i]
        if (window[0] > 0 and not EVENT):       # It's a new event
            EVENT = True
            if (len(indices) > 1):
                x = lookback(i, indices[-1])
            else:
                x = i-60
            indices.append(x)
        elif (window[0] > 0 and EVENT):         # It's an already marked event
            continue
        elif (window[0] < 1 and EVENT):         # It's an ending event
            EVENT = False
            while True:
                index, another = lookfar(i, md)
                if not another:
                    break
            indices.append(index)
        else:
            continue

    event_tups = []
    # Combine events that are too close
    indices.remove(363530)
    indices.remove(363489)
    indices.remove(401329)
    indices.remove(401284)
    for j in range(len(indices)-1):
        tup = tuple((indices[j], indices[j+1]))
        if not (j%2):
            event_tups.append(tup)

    for event in event_tups:
        labels = np.array(md[event[0]:event[1], 0], dtype='int')
        neg, pos = np.bincount(labels)
    return(indices, event_tups)

# Will adding 60 minutes encounter another event?
def lookfar(i, md):
    k = 0
    while (k < 60):
        k += 1
        if (md[k][0] > 0):
            while (md[k][0] > 0):
                k += 1
            return(i+k, True)
    return(i+60, False)

# Will looking back 60 minutes encounter the most recent event?
def lookback(i, e):
    b = i-60
    if (b<e):
        n = (b+e)//2    # Replace the old event ending with a halfway point
        return(n)
    return(b)

def get_nonevent_splits(event_tups, indices):
    for i in range(len(event_tups)-1):
        end_first = event_tups[i][1]
        begin_next = event_tups[i+1][0]
        difference = begin_next - end_first
        num_splits = difference // 10000
        for j in range(num_splits):
            index = 10000*j + end_first
            indices.append(index)
        i += 1
    indices.sort() 
    return(indices)

# Return complete list of index tuples
def combine_splits(indices):
    final = []
    for i in range(len(indices)-1):
        final.append((indices[i], indices[i+1]))
    final.sort()
    for f in final:
        if (f[0] == f[1]):
            final.remove(f)
    return(final)

# Generate a training set using all nonevent data and all event data except one event
def create_training_set(fv, md, event_tups, nonevents):
    # Randomly arrange the tuples of indices
    complete = event_tups
    for n in nonevents: 
        complete.append(n)
    random.shuffle(complete)
    
    # Restack blocks
    t = complete[0]
    init_fv = fv[t[0]:t[1]]
    init_md = md[t[0]:t[1]]
    complete.remove(t)
    for t in complete:
        f = fv[t[0]:t[1]]
        m = md[t[0]:t[1]]
        init_fv = np.vstack((init_fv, f))
        init_md = np.vstack((init_md, m))
    return(init_fv, init_md[:,0])

# Train the model using hardcoded hyperparameters from the optimal model configuration
def trainer(training_fv, training_labels, i, final_hp):
    run = "TestingEvent" + str(i)
    model = create_model(final_hp["neurons"], final_hp["input_dim"], final_hp["dropout"])
    trained = train_model(model, training_fv, training_labels, final_hp["epochs"], final_hp["batch"], final_hp["event_weight"], run)
    return(trained)

# Print the model's performance (metrics) on the testing data
def evaluate_model(model, test_fv, md_test, final_hp, r):
    scaler = StandardScaler()
    fv_test = scaler.fit_transform(test_fv)
    scores = model.evaluate(fv_test, md_test, verbose=0)
    print("Evaluating model:", file=open("eventlogs/EventLog"+str(r)+".txt", "a"))
    for i in range(len(scores)):
        print(model.metrics_names[i], scores[i], file=open("eventlogs/EventLog"+str(r)+".txt", "a"))



# Wrapper for model_testing.py functions
# Creates a CSV for the event with Actual Label, Predicted Label, Threshold Label, CW, Crac1 inlet, Crac2 inlet 
def make_predictions(trained_model, event, fv, md, r, final_hp):
    print("Begin predictions", file=open("eventlogs/EventLog"+str(r)+".txt", "a"))
    batch = final_hp["batch"]
    mostly(fv, md, event, r, trained_model, batch)
    print("End predictions", file=open("eventlogs/EventLog"+str(r)+".txt", "a"))



# Which event to exclude?
#event_start = int(sys.argv[1])
#event_end = int(sys.argv[2])
#r = int(sys.argv[3])
#event = tuple((event_start, event_end))

# Final hyperparameters
final_hp = {
    "epochs": 32,
    "batch": 16384,
    "dropout": 0.5,
    "neurons": 260,
    "input_dim": 260,
    "event_weight": 0.85}

# Read in complete datasets
fv = consolidator("fv", 5)
md = consolidator("md", 5)
print("First timestamp:", md[0,1])
print("Last timestamp:", md[-1,1])

# Locate events
#indices, events = get_event_splits()
#indices = get_nonevent_splits(events, indices)

# Add start and end indices to indices list
#indices.insert(0, 0)
#indices.append(463247)
#final = combine_splits(indices)
#nonevents = [i for i in final if not i in events]

# Remove laggy pump events
#del events[5:11]

# Train on all events except one, test on excluded event
#print("Event:", event, file=open("eventlogs/EventLog"+str(r)+".txt", "a"))
#event_tups = [e for e in events if e != event]
#training_fv, training_labels = create_training_set(fv, md, event_tups, nonevents)
#testing_fv = fv[event[0]:event[1]]
#testing_labels = md[event[0]:event[1],0]
#trained_model = trainer(training_fv, training_labels, r, final_hp)
#evaluate_model(trained_model, testing_fv, testing_labels, final_hp, r)
#make_predictions(trained_model, event, fv, md, r, final_hp)
#print("Event:", event, "- Completed.", file=open("eventlogs/EventLog"+str(r)+".txt", "a"))









