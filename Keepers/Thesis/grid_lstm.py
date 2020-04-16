import numpy as np
import pandas
import h5py
import sys
import time
import sys
import time

import scipy
import sklearn
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Given the path to an hdf5 file and a dataset to read
# Return a numpy array representing dataset in file
def read_hd5_file(path, dataset):
    hf = h5py.File(path, 'r')
    n1 = np.array(hf[dataset][:])
    return(n1)

def create_model():
    # Create Model
    model = Sequential()
    model.add(LSTM(8, input_shape=(5, 52), return_sequences=True, dropout=0.5, recurrent_dropout=0.0))
    model.add(LSTM(8, input_shape=(5, 52), return_sequences=False, dropout=0.5, recurrent_dropout=0.0))
    model.add(Dense(1, activation='sigmoid'))

    # Compile Model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return(model) 

t1 = time.time()
event_weight = float(sys.argv[1])
run = int(sys.argv[2])

print("Starting LSTM Grid Search - Run", run) 

# Fix random seed
np.random.seed(3)

# Read in and scale data
fv = read_hd5_file("Data/OrderedTrainingData.hdf5", "fv")
md = read_hd5_file("Data/OrderedTrainingData.hdf5", "md")[:,0]
scaler = StandardScaler()
fv = scaler.fit_transform(fv)
fv = np.reshape(fv, (len(fv), 5, 52))

# Set class weight
#The_Weight = class_weight.compute_class_weight('balanced', [0,1], md)
The_Weight = {0: 1-event_weight, 1: event_weight}

# Define parameter grid
batch_size = [16, 2048]
epochs = [1000]
param_grid = dict(batch_size=batch_size, epochs=epochs)

print("Passed the param grid checkpoint")

# Create the grid with 5 validation folds, use all cores
model = KerasClassifier(build_fn=create_model, verbose=0)
print("Passed the model checkpoint")
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=2, scoring=make_scorer(f1_score))
print("Passed the grid checkpoint")
grid_result = grid.fit(X=fv, y=md, shuffle=False, class_weight=The_Weight)
print("Passed the grid result checkpoint")

# Record grid_result in a JSON file
try:
    df = pandas.DataFrame(grid_result.cv_results_)
    hist_json = "Histories/LSTM/GridSearch/Run" + str(run) + "-history.json"
    with open(hist_json, "w") as f:
        df.to_json(f)
        f.close()
except Exception as e:
   print("Dataframe failed again.", e)

# Print Results
print("Best f1_score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print("Grid Search executed in ", time.time() - t1, "seconds\n")




