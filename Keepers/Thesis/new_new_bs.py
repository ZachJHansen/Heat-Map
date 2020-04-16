import datetime
import h5py
import numpy as np
import random

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

    print("Num Events:", len(event_tups))
    for event in event_tups:
        labels = np.array(md[event[0]:event[1], 0], dtype='int')
        neg, pos = np.bincount(labels)
        print("Start:", event[0], "- End:", event[1])
        print("Positive:", pos, "Negative:", neg)
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

# Min events: 0.1 * 9203 = 920
# Max events: 0.2 * 9203 = 1840
def check_conditions(candidates, md):
    event_dicts = examine_events(candidates, md[:,0])
    total = 0
    for d in event_dicts:
        total += d["Positive"]
    if (total < 1840 and total > 920):
        return(True, candidates)
    else:
        return(False, candidates)
            
# Create an array of dictionaries representing information about each event
def examine_events(events, labels):
    results = []
    for e in events:
        int_list = [int(i) for i in labels[e[0]:e[1]]]
        lbls = np.array(labels[e[0]:e[1]], dtype='int')
        neg, pos = np.bincount(lbls)
        event_data = {
            "Tuple": e,
            "Positive": pos,
            "Negative": neg
        }
        results.append(event_data)
    return(results)

def add_events(candidates, labels):
    event_dicts = examine_events(candidates, labels)
    while True:
        r = random.randrange(2,5)
        es = random.sample(event_dicts, k=r)
        total = 0
        for e in es:
            total += e["Positive"]
        if (total < 120 and total > 80):
            events = [i["Tuple"] for i in es]
            break
    return(events)
    
def get_size(tups):
    size = 0
    for t in tups:
        diff = t[1] - t[0]
        size += diff
    return(size)

def fracture(events, nonevents):
    event_len = len(events)
    nonevent_len = len(nonevents)
    events = list(set(events))
    nonevents = list(set(nonevents))
    if (event_len != len(events)):
        print("Events had redundant timestamps")
    if (nonevent_len != len(nonevents)):
        print("Nonevents had redundant timestamps")
    while True:
        random.shuffle(events)
        random.shuffle(nonevents)
        r1 = random.randrange(2, 5)
        r2 = random.randrange(2, 5)
        test_candidates = events[0:r1]
        success, testing_tuples = check_conditions(test_candidates, md)
        if not success:
            continue
        else:
            val_cands = events[r1:r2]
            success, validation_tuples = check_conditions(val_cands, md)
            if not success:
                continue
            else:
                for i in range(6):
                    testing_tuples.append(nonevents[i])
                    validation_tuples.append(nonevents[i+6])
                training_tuples = nonevents[11:]
                for event in events[r2:]:
                    training_tuples.append(event)
                break
 
    print("Redundants")
    for t in validation_tuples:
        if t in testing_tuples:
            print("Crap:", t)
            validation_tuples.remove(t)
    for t in training_tuples:
        if (t in validation_tuples or t in testing_tuples):
            print("Double crap:", t)
            training_tuples.remove(t)

    random.shuffle(testing_tuples)
    random.shuffle(training_tuples)
    random.shuffle(validation_tuples)
    
    print("\nTesting Tuples")
    for t in testing_tuples:
        print(t)
    print("Validation Tuples")
    for t in validation_tuples:
        print(t)
    print("Training Tuples")
    for t in training_tuples:
        print(t)

    total = [t for t in training_tuples]
    for i in testing_tuples:
        total.append(i)
    for j in validation_tuples:
        total.append(j)
    new_total = list(set(total)).sort()    

    jenga(total, "total")
    jenga(testing_tuples, "test")
    jenga(validation_tuples, "val")
    jenga(training_tuples, "train")

# Stack blocks on top of each other to form block shuffled matrices
def jenga(tupleware, path):
    # Read in all data
    fv = consolidator("fv", 5)
    md = consolidator("md", 5)
    
    # Restack fv blocks
    t = tupleware[0]
    init_fv = fv[t[0]:t[1]]
    init_md = md[t[0]:t[1]]
    tupleware.remove(t)
    for t in tupleware:
        f = fv[t[0]:t[1]]
        m = md[t[0]:t[1]]
        init_fv = np.vstack((init_fv, f))
        init_md = np.vstack((init_md, m))
    print("FV Shape:", init_fv.shape)
    print("MD Shape:", init_md.shape)

    # Write to new files
    hf = h5py.File("JOB.h5", 'a')
    hf.create_dataset("fv_"+path, data=init_fv)
    hf.create_dataset("md_"+path, data=init_md)
    hf.close()

# There are certain timestamps representing yellow zones, not full blown overheating events
# We are discarding those for now
def remove_select(events, md):
    # Discarding all events that occur in the timeframe: 2019-08-13 09:06:28 through 2019-08-28 07:29:15 CDT
    # 1565687188 - 1566977355 
    for event in events:
        start = md[event[0], 1]
        end = md[event[1], 1]
        if (start > 1565687188 and end < 1566977355):
            events.remove(event)
            print(event)
    return(events)

# There are a few big "events" that really don't look like events.
# They occur August through September, and they really don't resemble any of the other marked events.
# They lack the massive crac spike that i think are the main indicators of an overheating event.
# Unfortunately, i think block shuffle selected them as the ONLY events in my train/validation data.
# Which i think explains the 0 precision, 0 recall, 0 fp (aka the model never said "event") in the validation data
# Because it learned on a very different type of event. In fact, i wish we hadn't marked the big low temperature time spans.
# So f*ck it, lets discard them


#md = consolidator("md", 5)
#start = md[0,1]
#end = md[-1,1]
#print("Start:", start)
#print("End", end)

md = consolidator("md", 5)
indices, events = get_event_splits()
indices = get_nonevent_splits(events, indices)
# Add start and end indices to indices list
indices.insert(0, 0)
indices.append(463247)
final = combine_splits(indices)
nonevents = [i for i in final if not i in events]
#del events[5:11] # Remove laggy pump events
fracture(events, nonevents)
train = read_hd5_file("JOB.h5", "md_train")
validate = read_hd5_file("JOB.h5", "md_val")
test = read_hd5_file("JOB.h5", "md_test")

labels = np.array(train[:,0], dtype='int')
neg, pos = np.bincount(labels)
print("\nTraining Labels")
print("Positive:", pos, "Negative:", neg, "Total:", neg+pos, "Pos Percent:", round((pos/(neg+pos)), 4))
labels = np.array(validate[:,0], dtype='int')
neg, pos = np.bincount(labels)
print("Validation Labels")
print("Positive:", pos, "Negative:", neg, "Total:", neg+pos, "Pos Percent:", round((pos/(neg+pos)), 4))
labels = np.array(test[:,0], dtype='int')
neg, pos = np.bincount(labels)
print("Testing Labels")
print("Positive:", pos, "Negative:", neg, "Total:", neg+pos, "Pos Percent:", round((pos/(neg+pos)), 4))







