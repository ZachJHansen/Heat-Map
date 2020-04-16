import scipy
import numpy as np
import pandas
import sklearn
import h5py
import sys
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.metrics import *

from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Given the path to an hdf5 file and a dataset to read
# Return a numpy array representing dataset in file
def read_hd5_file(path, dataset):
    hf = h5py.File(path, 'r')
    n1 = np.array(hf[dataset][:])
    return(n1)

def create_model(neurons, in_dim, dropout, recur_drop):
    METRICS = [TruePositives(name='tp'), FalsePositives(name='fp'), TrueNegatives(name='tn'), FalseNegatives(name='fn'), BinaryAccuracy(name='accuracy'), Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]

    # Define model
    model = Sequential()
    # See these links for help with input shapes:
#https://github.com/MohammadFneish7/Keras_LSTM_Diagram
#https://stats.stackexchange.com/questions/274478/understanding-input-shape-parameter-in-lstm-with-keras
#https://stackoverflow.com/questions/48140989/keras-lstm-input-dimension-setting

    model.add(LSTM(neurons, input_shape=(5, 52), return_sequences=True, dropout=dropout, recurrent_dropout=recur_drop))
    model.add(LSTM(neurons, input_shape=(5, 52), dropout=dropout, recurrent_dropout=recur_drop))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)
    return(model)

def train_model(model, fv_train, md_train, fv_val, md_val, epochs, batch, run, event_weight):
    print("\nFV Train Shape:", fv_train.shape, file=open("LSTMTrainingLog.txt", "a"))
    print("FV Validation Shape:", fv_val.shape, file=open("LSTMTrainingLog.txt", "a"))

    #The_Weight = class_weight.compute_class_weight('balanced', [0,1], md_train)
    The_Weight = {0: 1 - event_weight, 1: event_weight} 
    print("Run", run, ": Starting model fitting. ", file=open("LSTMTrainingLog.txt", "a"))
    try:
        hist = model.fit(x=fv_train, y=md_train, batch_size=batch, epochs=epochs, verbose=0, validation_data=(fv_val, md_val), class_weight=The_Weight, shuffle=False)
        print("Run", run, "Model training successful. ", file=open("LSTMTrainingLog.txt", "a"))
        model.save("Models/LSTM/model" + run + ".h5")
        # Record training history in a JSON file
        df = pandas.DataFrame(hist.history)
        hist_json = "Histories/LSTM/Run" + run + "-history.json"
        with open(hist_json, "w") as f:
            df.to_json(f)
            f.close()
        return(model)
    except Exception as e:
        print("Run", run, "Something's gone wrong during model training:", e, file=open("LSTMTrainingLog.txt", "a"))
        return(None)


# Hyperparameters
neurons = int(sys.argv[1])
batch_size = int(sys.argv[2])
epochs = int(sys.argv[3])
dropout = float(sys.argv[4])
recur_drop = float(sys.argv[5])
validation_split = float(sys.argv[6])
event_weight = float(sys.argv[7])
run = sys.argv[8]

t0 = time.time()

scaler = StandardScaler()
fv_train = scaler.fit_transform(read_hd5_file("Data.h5", "fv_train"))
md_train = read_hd5_file("Data.h5", "md_train")
fv_val = scaler.fit_transform(read_hd5_file("Data.h5", "fv_val"))
md_val = read_hd5_file("Data.h5", "md_val")

in_dim = 260 #len(fv[0])

print("\nMany Epochs Run", run, ":\n", file=open("LSTMTrainingLog.txt", "a"))
fv_train = np.reshape(fv_train, (len(fv_train), 5, 52))
fv_val = np.reshape(fv_val, (len(fv_val), 5, 52))
model = create_model(neurons, in_dim, dropout, recur_drop)
trained_model = train_model(model, fv_train, md_train, fv_val, md_val, epochs, batch_size, run, event_weight)

t3 = time.time()
print("Many Epochs Run", run, "completed in time:", t3 - t0, file=open("LSTMTrainingLog.txt", "a"))













