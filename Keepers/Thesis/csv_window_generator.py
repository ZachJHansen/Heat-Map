# Barrett paper had some recommendations for dealing with mixed
# labels in your window but that may have been for regression




# Warning! Looks broken


import h5py
import numpy as np
import datetime
from pymongo import MongoClient
import pymongo
import time


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

# Add N-1 minutes to datetime, return result
def increment(time, N):
#    next_time = time + datetime.timedelta(minutes=N)
    next_time = time + (N-1) * 60 
    return(next_time)

# I added an index to the FeatureVectors collection for Timestamp
def retrieve_data(ip, port, username, password, src, database, collection, start_date, end_date):
    data = []
    client = MongoClient(ip, port)
    client.admin.authenticate(username, password, mechanism='SCRAM-SHA-1', source=src)
    db = client[database]
    coll = db[collection]
    cursor = coll.find({"$and": [{"Timestamp": {'$gte': start_date}}, 
                          {"Timestamp": {'$lte': end_date}}]}, no_cursor_timeout=True).sort("Timestamp", pymongo.ASCENDING)
   # cursor = coll.find({}, no_cursor_timeout=True).sort("Timestamp", pymongo.ASCENDING)
    # featurevector_matrix = [upsOnBattery, ..., Rack19RelativeHumidity-4] 
    # metadata_matrix = [Label, timestamp, gap_flag]
    doc = cursor[0]
    fv = doc["Feature_Vector"]
    fv_matrix = np.array(fv[1:])
    md_matrix = np.array([fv[0], epochify(doc["Timestamp"]), 0])
    temp = cursor.next()
    i = 1
    jagged_fv = len(fv_matrix)
    jagged_md = len(md_matrix)
    print(jagged_fv)
    print(fv[1:])
    print(len(fv[1:]))
    while (True):
        i += 1
        try:
            doc = cursor.next()
            fv = doc["Feature_Vector"]
            ts = epochify(doc["Timestamp"])
            if (len(fv[1:]) != jagged_fv):
                print("Attempting to create jagged feature vector matrix with array of length ", len(fv), " at timestamp ", ts, " - Skipped.")
            else: 
                fv_matrix = np.vstack((fv_matrix, fv[1:]))
                md_matrix = np.vstack((md_matrix, [fv[0], epochify(doc["Timestamp"]), 0]))
        except StopIteration:
            break
        except Exception as e:
            print("Error: ", e)
    cursor.close()
    print("Records: ", i)
    return((fv_matrix, md_matrix))

# Convert datetime object to equivalent seconds since epoch
def epochify(timestamp):
    t = (timestamp - datetime.datetime(1970,1,1)).total_seconds()
    return(t)


def write_partial_file(start_date, end_date, file_name):
    print("Starting data retrieval for ", file_name)
    t1 = time.time()
    data = retrieve_data('10.1.6.102', 27017, 'logger', '3mpZjGdS', 'monitoring', 'monitoring', 'FeatureVectors', start_date, end_date)
    print("Time to retrieve data: ", time.time() - t1)

    N = 5
    L = len(data[0])
    t2 = time.time()
    mx_tuple = generate_window(N, 0, data[0], data[1])
    fv_mx = np.array(mx_tuple[0])
    md_mx = np.array(mx_tuple[1])
    f_len = len(fv_mx)
    m_len = len(md_mx)
    print("Lengths")
    print(f_len)
    print(m_len)
    for i in range(1, (L-N+1)):
        try:
            window = generate_window(N, i, data[0], data[1])
            if (len(window[0]) != f_len or len(window[1]) != m_len):
                print("Jagged matrix at timestamp: ", str(window[1][1]))
            else:
                fv_mx = np.vstack((fv_mx, window[0]))
                md_mx = np.vstack((md_mx, window[1]))
        except Exception as e:
            print("Error: ", e)
            print("Length of window " + str(i) + str(len(window)))
    print("Time to generate windows: ", time.time() - t2)    
    
    ones = 0
    zeros = 0
    for row in md_mx:
        if row[0] == 1: ones += 1
        else: zeros += 1


    t3 = time.time()
    fv_file_name = file_name + "_fv_mx" + ".csv"
    md_file_name = file_name + "_md_mx" + ".csv"
    np.savetxt(fv_file_name, fv_mx, fmt='%3.2f', delimiter=",")
    np.savetxt(md_file_name, md_mx, fmt='%10.0f', delimiter=",") 
    print("Length of matrix: ", len(fv_mx))
    print("Number of 1 labels: ", ones)
    print("Length of 0 labels: ", zeros, "\n")



# Main
N = 5
start_date = datetime.datetime(2018, 12, 11, 16, 3, 0, 0)
end_date = datetime.datetime(2019, 11, 30, 0, 0, 0, 0)
file_name = "total"
write_partial_file(start_date, end_date, file_name)

ready = False
if ready:
    for mon in ["feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct"]:
        start_date = end_date - datetime.timedelta(minutes=N) + datetime.timedelta(minutes=2) 
        end_date = end_date + datetime.timedelta(days=31)
        try:
            write_partial_file(start_date, end_date, mon+".hdf5")
        except Exception as e:
            print("Error on month ", mon)
            print("Exception: ", e)

# If you are going to shuffle windows, do it before you split the matrix
# That way fv_mx and md_mx are in the same order
# Or recombine them in chiron.py and shuffle them there
# Just don't split and then shuffle seperately
#k = 76*N
#fv_mx = Matrix[:,:k]
#md_mx = Matrix[:,k:]














