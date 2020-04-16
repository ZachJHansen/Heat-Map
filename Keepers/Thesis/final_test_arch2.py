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

def evaluate_model(model, test_fv, md_test):
    scaler = StandardScaler()
    fv_test = scaler.fit_transform(test_fv)
    scores = model.evaluate(fv_test, md_test, verbose=0)
    for i in range(len(scores)):
        print(model.metrics_names[i], scores[i], file=open("Testing_Arch2.txt", "a"))

# Hyperparameters
#best_model = sys.argv[1]

# Read in test data
#test_fv = read_hd5_file("../Data/BST.h5", "fv_test")
#test_md = read_hd5_file("../Data/BST.h5", "md_test")[:,0]
test_dates = read_hd5_file("../Data/BST.h5", "md_test")[:,1]
print(test_dates)
# Train model
#print("Final Testing: Arch2", file=open("Testing_Arch2.txt", "a"))
#print("Testing FV Shape -", test_fv.shape)
#print("Testing MD Shape -", test_md.shape)
#model = load_model(best_model)
#evaluate_model(model, test_fv, test_md)
#print("Testing Complete")

