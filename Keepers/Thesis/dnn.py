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
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold

#https://stackoverflow.com/questions/41246293/how-do-i-train-a-neural-network-in-keras-on-data-stored-in-hdf5-files
#https://keras.io/models/sequential/
#https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
#https://machinelearningmastery.com/cross-entropy-for-machine-learning/

# Given the path to an hdf5 file and a dataset to read
# Return a numpy array representing dataset in file
def read_hd5_file(path, dataset):
    hf = h5py.File(path, 'r')
    n1 = np.array(hf[dataset][:])
    return(n1)

def create_model(layer_width, depth, in_dim):
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
    model.add(Dense(layer_width, input_dim = in_dim, activation = 'relu'))
    model.add(Dropout(0.2, noise_shape=None, seed=SEED))
    model.add(Dense(layer_width, activation = 'relu'))
    model.add(Dropout(0.2, noise_shape=None, seed=SEED))
    model.add(Dense(layer_width, activation = 'relu'))
    model.add(Dropout(0.2, noise_shape=None, seed=SEED))
    model.add(Dense(layer_width, activation = 'relu'))
    model.add(Dropout(0.2, noise_shape=None, seed=SEED))
    model.add(Dense(layer_width, activation = 'relu'))
    model.add(Dropout(0.2, noise_shape=None, seed=SEED))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)
    return(model)

def train_model_kfold(model, fv, md, train, test, epochs, batch, run):
    # Fit model
    scaler = StandardScaler()
    fv_train = scaler.fit_transform(fv[train])
    md_train = md[train]
    fv_test = scaler.transform(fv[test])
    md_test = md[test]
    print("FV Train Shape:", fv_train.shape, file=open("BigBatchHistory.txt", "a"))
    print("MD Train Shape:", md_train.shape, file=open("BigBatchHistory.txt", "a"))
    print("FV Validation Shape:", fv_test.shape, file=open("BigBatchHistory.txt", "a"))
    print("MD Validation Shape:", md_test.shape, file=open("BigBatch1History.txt", "a"))
    The_Weight = {0: 0.25, 1: 0.75} 
    #The_Weight = class_weight.compute_class_weight('balanced', [0,1], md_train)
    print("Run", run, ": Starting model fitting. ", file=open("BigBatchHistory.txt", "a"))
    try:
#        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        hist = model.fit(x=fv_train, y=md_train, batch_size=batch, epochs=epochs, verbose=0, shuffle=False, class_weight=The_Weight)
        print("Run", run, "Model training successful. ", file=open("BigBatchHistory.txt", "a"))
        
        # Evaluate the model
        scores = model.evaluate(fv_test, md_test, verbose=0)
        for i in range(len(scores)):
            print(model.metrics_names[i], scores[i], file=open("BigBatchHistory.txt", "a"))

        # Save the model
        model.save("Models/DNN/model" + run + ".h5")

        # Record training history in a JSON file
        df = pandas.DataFrame(hist.history)
        hist_json = "Histories/DNN/Run" + run + "-history.json"
        with open(hist_json, "w") as f:
            df.to_json(f)
            f.close()
        return(model)
    except Exception as e:
        print("Run", run, "Something's gone wrong during model training:", e, file=open("BigBatchHistory.txt", "a"))
        return(None)

def train_model(model, fv, md, train, epochs, batch, run):
    scaler = StandardScaler()
    fv_train = scaler.fit_transform(fv[train:,])
    md_train = md[train:,]
    fv_val = scaler.transform(fv[:train,])
    md_val = md[:train,]
    print("FV Train Shape:", fv_train.shape)
    print("MD Train Shape:", md_train.shape)
    #The_Weight = {0: 0.02, 1: 0.98}
    print("FV Validation Shape:", fv_val.shape)
    print("MD Validation Shape:", md_val.shape)
    The_Weight = {0: 0.25, 1: 0.75}
#The_Weight = class_weight.compute_class_weight('balanced', [0,1], md_train)
    print("Run", run, ": Starting model fitting. ", file=open("BigBatchHistory.txt", "a"))
    try:
    #    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)
        hist = model.fit(x=fv_train, y=md_train, batch_size=batch, epochs=epochs, verbose=0, validation_data=(fv_val, md_val), shuffle=False, class_weight=The_Weight)
        print("Run", run, "Model training successful. ", file=open("BigBatchHistory.txt", "a"))

        # Save the model
        model.save("Models/DNN/model" + run + ".h5")

        # Record training history in a JSON file
        df = pandas.DataFrame(hist.history)
        hist_json = "Histories/DNN/Run" + run + "-history.json"
        with open(hist_json, "w") as f:
            df.to_json(f)
            f.close()
        return(model)
    except Exception as e:
        print("Run", run, "Something's gone wrong during model training:", e, file=open("BigBatchHistory.txt", "a"))
        return(None)


# Hyperparameters
layer_width = int(sys.argv[1])
depth = int(sys.argv[2])
epochs = int(sys.argv[3])
batch_size = int(sys.argv[4])
run = sys.argv[5]

t0 = time.time()

if (False):
    fv = read_hd5_file("CompleteTrainingData.h5", "fv")
    md = read_hd5_file("CompleteTrainingData.h5", "md")[:,0]
    in_dim = len(fv[0])
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=9)
    model = create_model(layer_width, depth, in_dim)
    model.summary()
    for train, test in kfold.split(fv, md):
        #model = load_model("Models/DNN/model18.h5")
        trained_model = train_model_kfold(model, fv, md, train, test, epochs, batch_size, run)
elif (True):
    print("Timeseries K-fold: Run 32")
    fv = read_hd5_file("Data/OrderedTrainingData.hdf5", "fv")
    md = read_hd5_file("Data/OrderedTrainingData.hdf5", "md")[:,0]
    in_dim = len(fv[0])
    print("Input Dimension:", in_dim)
    model = create_model(layer_width, depth, in_dim)
    model.summary()
    tscv = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in tscv.split(fv):
        trained_model = train_model_kfold(model, fv, md, train_index, test_index, epochs, batch_size, run)
else:
    fv = read_hd5_file("Data/OrderedTrainingData.hdf5", "fv")
    md = read_hd5_file("Data/OrderedTrainingData.hdf5", "md")[:,0]
    in_dim = len(fv[0])
    #train_split = round(len(fv)*0.8)
    model = create_model(layer_width, depth, in_dim)
    #labels = np.array(md, dtype='int')
    val_split = round(len(md)*0.2)
    #print(val_split)
    #training_labels = labels[val_split:]
    #testing_labels = labels[:val_split]
    #neg, pos = np.bincount(training_labels)
    #total = neg + pos
    #print('Training Labels:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total)) 
    #neg, pos = np.bincount(testing_labels)
    #total = neg + pos
    #print('Testing Labels:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))  
    model.summary()
    trained_model = train_model(model, fv, md, val_split, epochs, batch_size, run)

t3 = time.time()
print("DNN Run", run, "completed in time:", t3 - t0)

















