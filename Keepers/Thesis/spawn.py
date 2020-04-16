import subprocess
import datetime
import multiprocessing as mp
import week_window_gen as wwg
import time

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

# Given a window length, generate array of tuples
# Tuples contain start datetime, end datetime, one day seperation
def date_gen_hour(N):
    dates = []
    # Initialize first hour
    start_date = datetime.datetime(2019, 1, 2, 21, 27, 0, 0)
    end_date = datetime.datetime(2019, 1, 2, 22, 27, 0, 0)
    tupleware = (start_date.strftime("%Y-%m-%d %H:%M:%S"), end_date.strftime("%Y-%m-%d %H:%M:%S"))
    dates.append(tupleware)
    while (True):
        start_date = end_date - datetime.timedelta(minutes=N) + datetime.timedelta(minutes=2)
        end_date = end_date + datetime.timedelta(hours=1)
        tupleware = (start_date.strftime("%Y-%m-%d %H:%M:%S"), end_date.strftime("%Y-%m-%d %H:%M:%S"))
        dates.append(tupleware)
        if (end_date >= datetime.datetime(2019, 1, 3, 21, 27, 0, 0)):
            break
    return(dates)

# Main
if __name__=="__main__":
    dates = date_gen_week(5)
    t1 = time.time()
    #for t in dates:
    #    print(t[0], " | ", t[1])
    pool = mp.Pool(processes=mp.cpu_count())
    result = pool.imap_unordered(wwg.grand_poobah, dates)
    pool.close()
    pool.join()
    for i in result:
        print(i)
    print("Time to execute: ", time.time() - t1)




