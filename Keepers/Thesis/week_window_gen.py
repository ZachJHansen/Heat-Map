import h5py
import numpy as np
import datetime
from pymongo import MongoClient
import pymongo
import time
import sys

# Fetch all records from FeatureVectors collection 
# with timestamps in between start and end dates, inclusive
# Each record creates a row in the matrices
# Return tuple (feature vector matrix, metadata matrix)
def fetch(start, end):
    start_date = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")

    client = MongoClient('10.1.6.102', 27017)
    client.admin.authenticate('logger', '3mpZjGdS', mechanism='SCRAM-SHA-1', source='monitoring')
    db = client['monitoring']
    coll = db['FeatureVectors']
    cursor = coll.find({"$and": [{"Timestamp": {'$gte': start_date}},
                          {"Timestamp": {'$lte': end_date}}]}, no_cursor_timeout=True).sort("Timestamp", pymongo.ASCENDING)

    # First row in matrix has to be dtype float64, not object
    i = 0
    while (True):
        doc = cursor[i]
        fv = doc["Feature_Vector"]
        fv_matrix = np.array(fv[1:])
        md_matrix = np.array([fv[0], epochify(doc["Timestamp"]), 0])
        if (fv_matrix.dtype != "float64"):
            print("Missing values in feature vector at: ", doc["Timestamp"], " Skipped.", file=open("ErrorLog.txt", "a"))
            i += 1
        else:
            jagged_fv = len(fv_matrix)
            jagged_md = len(md_matrix)
            break
    #print(fv_matrix)
    #print("Shape:", fv_matrix.shape)
    #print("Dtype:", fv_matrix.dtype)
    # Pile on the rows, skipping them if they are of the wrong shape or dtype
    while (True):
        i += 1
        try:
            doc = cursor[i]
            fv1 = doc["Feature_Vector"]
            ts = epochify(doc["Timestamp"])
            if (len(fv1[1:]) != jagged_fv):
                print("Attempting to create jagged feature vector matrix with array of length ", len(fv), " at timestamp ", ts, " - Skipped.", file=open("ErrorLog.txt", "a"))
            else:
                fv = np.array(fv1[1:])
                if (fv.dtype != 'float64'):
                    print("Missing values in feature vector at: ", doc["Timestamp"], " Skipped.", file=open("ErrorLog.txt", "a"))
                else:
                    fv_matrix = np.vstack((fv_matrix, fv))
                    md_matrix = np.vstack((md_matrix, [fv1[0], epochify(doc["Timestamp"]), 0]))
        except IndexError:
            break
        except Exception as e:
            print("Error while fetching data during interval: ", start, " Error: ", e, file=open("ErrorLog.txt", "a"))
    cursor.close()
    print(str(i), " records retrieved from interval ", start, file=open("performanceLog.txt", "a"))
    return((fv_matrix, md_matrix))


# Returns a tuple of 2 2D numpy arrays (feature vector and metadata matrices)
# Each row in fv_mx is a window (N concatenated feature vectors)
# Each row in md_mx is metadata (label, timestamp, gapflag) for corresponding winow
def window(fv, md, N):
    L = len(fv)
    mx_tuple = generate_window(N, 0, fv, md)
    fv_mx = np.array(mx_tuple[0])
    md_mx = np.array(mx_tuple[1])
    f_len = len(fv_mx)
    m_len = len(md_mx)
    for i in range(1, (L-N+1)):
        try:
            window = generate_window(N, i, fv, md)
            if (len(window[0]) != f_len or len(window[1]) != m_len):
                print("Jagged matrix at timestamp: ", str(window[1][1]), file=open("ErrorLog.txt", "a"))
            else:
                fv_mx = np.vstack((fv_mx, window[0]))
                md_mx = np.vstack((md_mx, window[1]))
        except Exception as e:
            print("Error while generating window after timestamp: ", md_mx[i-1][1], " Error: ", e, file=open("ErrorLog.txt", "a"))
    i += 1
    print(str(i), " windows generated during interval starting at seconds since epoch:", md[0][1], file=open("performanceLog.txt", "a"))
    return((fv_mx, md_mx))

# Generate a window of (N) records, starting from an index
# Add some sort of metadata if there is some skipped timestamp
def generate_window(N, index, fv_mx, metadata_mx):
    label_max = 0

    # Does the window have missing data?
    gap_flag = 0
    start_time = metadata_mx[index][1]
    end_time = metadata_mx[index+N-1][1]
    if (increment(start_time, N) < end_time):
        gap_flag = 1

    window = np.array(fv_mx[index])
    label = metadata_mx[index][0]
    label_max = max(label_max, label)
    for i in range(1,N):
        window = np.concatenate((window, fv_mx[index+i]), axis=0)
        label = metadata_mx[index+i][0]
        label_max = max(label_max, label)

    # Window is a tuple (input_vector, metadata_vector)
    Window = (window, [label_max, start_time, gap_flag])
    return(Window)

# Convert datetime object to equivalent seconds since epoch
def epochify(timestamp):
    t = (timestamp - datetime.datetime(1970,1,1)).total_seconds()
    return(t)

# Add N-1 minutes to datetime, return result
def increment(time, N):
    next_time = time + (N-1) * 60
    return(next_time)

# Write the matrix of windows to a CSV file
# Filename is start_date-end_date.csv
def csv_write(start, end, fv_mx, md_mx):
    d1 = start.split(" ")[0]
    d2 = end.split(" ")[0]
    fv_filename = "fv_" + d1 + "-" + d2 + ".csv"
    md_filename = "md_" + d1 + "-" + d2 + ".csv"
    np.savetxt(fv_filename, fv_mx, fmt='%3.2f', delimiter=",")
    np.savetxt(md_filename, md_mx, fmt='%10.0f', delimiter=",") 

def hdf_write(start, end, fv_mx, md_mx):
    d1 = start.split(" ")[0]
    d2 = end.split(" ")[0]
    filename = d1 + "-" + d2 + ".hdf5"
    hf = h5py.File(filename, 'w')
    hf.create_dataset("fv", data=fv_mx)
    hf.create_dataset("md", data=md_mx)
    hf.close()

# Main
def grand_poobah(dates):
    start_date = dates[0]
    end_date = dates[1]

    # Retrieve records
    success = False
    try:
        matrices = fetch(start_date, end_date)
        fv_mx = matrices[0]
        md_mx = matrices[1]
        if (len(fv_mx) > 0 and len(md_mx) > 0):
            success = True
    except Exception as E1:
        print("Couldnt retrieve records starting at ", start_date, file=open("ErrorLog.txt", "a"))
        print("Experienced exception E1: ", E1, "\n", file=open("ErrorLog.txt", "a"))    

    # Window records
    if success:
        success = False
        try:
            windows = window(fv_mx, md_mx, 5)
            fv_mx = windows[0]
            md_mx = windows[1]
            if (len(fv_mx) > 0 and len(md_mx) > 0):
                success = True
        except Exception as E2:
            print("Couldnt window records starting at ", start_date, file=open("ErrorLog.txt", "a"))
            print("Experienced exception E2: ", E2, "\n", file=open("ErrorLog.txt", "a"))  

    # Write to HDF5 file
    if success:
        try:
            hdf_write(start_date, end_date, fv_mx, md_mx)
        except Exception as E3:
            print("Couldnt write to hdf5 file starting at ", start_date, file=open("ErrorLog.txt", "a"))
            print("Experienced exception E3: ", E3, "\n", file=open("ErrorLog.txt", "a"))  

    return(success)



















 







