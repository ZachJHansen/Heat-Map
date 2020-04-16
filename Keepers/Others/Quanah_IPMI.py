from pymongo import MongoClient
from threading import Thread 
from queue import Queue
from multiprocessing import Pool
import database
import datetime
import subprocess
import sys

#Get the path of the list of IPs to monitor
path_list_of_IPs = sys.argv[1]

# For a given IP address, fetch the system event log and store it in a queue, along with a type ("sel"), the IP address, and the compute node name
# In the event of an error, store the error in an error queue
def get_sel(ip, data_q, error_q):
    try:
        sel = subprocess.check_output('ipmitool -I lanplus -H ' + ip[0] + ' -U root -P nivipnut sel list', shell=True, stderr=subprocess.STDOUT).decode('ascii')
        data_q.put(("sel",sel,ip[0],ip[1]))
    except Exception as error:
        error_q.put(error)

# For a given IP address, fetch the sdr and store it in a queue, along with a type ("sdr"), the IP address, and the compute node name
# In the event of an error, store the error in an error queue
def get_sdr(ip, data_q, error_q):
    try:
        sdr = subprocess.check_output('ipmitool -I lanplus -H ' + ip[0] + ' -U root -P nivipnut sdr elist full | egrep -v "Disabled|No Reading|Usage|Current|Voltage"', shell=True, stderr=subprocess.STDOUT).decode('ascii')
        data_q.put(("sdr",sdr,ip[0],ip[1]))
    except Exception as error:
        error_q.put(error)

# For a given IP address, fetch the power status and store it in a queue, along with a type ("pwr_stat"), the IP address, and the compute node name
# In the event of an error, store the error in an error queue
def get_power_status(ip, data_q, error_q):
    try:
        pwr_stat = subprocess.check_output('ipmitool -I lanplus -H ' + ip[0] + ' -U root -P nivipnut power status', shell=True, stderr=subprocess.STDOUT).decode('ascii')
        data_q.put(("pwr_stat",pwr_stat,ip[0],ip[1]))
    except Exception as error:
        error_q.put(error)

# For a given (IP, Hostname) tuple, create a thread for each ipmitool command.
# Returns a list of the response strings from each command (does not include errors)
# Errors are printed to an error log
def get_node_data(ip):
    if ping_node(ip[0]):
        results = []
        threads = []
        data_q = Queue()
        error_q = Queue()

        th1 = Thread(target = get_sel, args = (ip, data_q, error_q,))
        th1.start()
        threads.append(th1)

        th2 = Thread(target = get_sdr, args = (ip, data_q, error_q,))
        th2.start()
        threads.append(th2)

        th3 = Thread(target = get_power_status, args = (ip, data_q, error_q,))
        th3.start()
        threads.append(th3)

        for thread in threads:
            thread.join()

        while not data_q.empty():
            results.append(data_q.get())

        while not error_q.empty():
            error = error_q.get()
            print("QUANAH_IPMI ERROR: %s at time: %s" % (error, datetime.datetime.utcnow()))

        return parseResults(results)
    else:
       print("QUANAH_IPMI ERROR: Unable to connect to (Host: %s, IP: %s) at time: %s" % (ip[1], ip[0], datetime.datetime.utcnow()))
       return None

# The result_list is a list of tuples for one IP address. Each tuple contains a type (sel, sdr, or pwr),
# a string representing the response for the ipmitool command corresponding to the type, and an (IP address, hostname) tuple
# If the list is not empty, the response strings will be parsed and stored in record
def parseResults(result_list):
    record = {}
    try:
        if result_list:
            record["IP Address"] = result_list[0][2].strip()
            record["Hostname"] = result_list[0][3].strip()
            for result in result_list:
                if (result[0] == "sel"):
                    parsed = parseSEL(result[1])
                    record[parsed[0]] = parsed[1]
                elif (result[0] == "sdr"):
                    parsed = parseSDR(result[1])
                    for pair in parsed:
                        key = pair[0]
                        value = pair[1]
                        if (key == "Input Voltage"):
                            record[key] = value
                        else:
                            record[key] = int(value)
                else:
                    parsed = parsePWR(result[1])
                    record[parsed[0]] = parsed[1]
            record["Timestamp"] = datetime.datetime.utcnow()    
    except Exception as error:
        print("QUANAH_IPMI ERROR: Error while parsing host {%s, %s}: Error: %s at %s" % (result_list[0][2], result_list[0][3], error, datetime.datetime.utcnow()))
    return record

# A non-empty SEL is indicative of trouble
def parseSEL(response):
    lines = response.split("\n")
    length = len(lines) - 1    
    if (length == 1):
        if ("Log area reset/cleared" in lines[0]):
            return (("SEL Line Count", 0))
    return (("SEL Line Count", length))

# Accepts a response string
# Returns a list of (key, value) tuples to be stored in a dictionary
def parseSDR(response):
    results = []
    split_resp = response.split("\n")
    for line in split_resp:
        if line:
            list1 = line.split("|")
            key = list1[0].strip()
            if (key == "Temp"):
                key = key + list1[1].strip()
#                if (list1[1] == " 0Eh "):
#                    key = "CPU1 Temp"
#                else:
#                    key = "CPU2 Temp"
            val = list1[4].split(" ")[1]
            results.append((key,float(val)))
    return results

# Accepts a response string
# Returns a tuple (key, value)
def parsePWR(response):
    split_resp = response.split(" ")
    N = len(split_resp)
    val = split_resp[N-1].split("\n")[0]
    return (("Chassis Power Status", val))

# Creates a list of all (IP addresses, names) of interest from a file
def generateHosts(file_name):
    host_list = []
    with open(file_name) as file:
        lines = [line.rstrip('\n') for line in file]
        for line in lines:
            if line:
                temp = line.split("            ")
                ip = temp[0]
                name = temp[len(temp)-1]
                host_list.append((ip, name))
    return host_list

# Check if a given node is responsive
def ping_node(ip):
    try:
        ping_status = subprocess.check_output("ping -c 1 -W 1.5 " + ip, shell=True).decode("ascii")
        return True
    except Exception as error:
    #    print("QUANAH_IPMI ERROR: Ping encountered exception {%s} for IP %s at time %s." % (error, ip, datetime.datetime.utcnow()))
        return False

# Main
if __name__=="__main__":
    hosts = generateHosts(path_list_of_IPs)
    
    pool = Pool(256)
    results = [pool.apply_async(get_node_data, (host,)) for host in hosts]
    pool.close()
    pool.join()

    success = database.dump('MP_Array', '10.1.6.102', 27017, 'logger', '3mpZjGdS', 'monitoring', 'monitoring', 'IPMI', results)
    if not success: 
        print("QUANAH_IPMI ERROR: Failed to record data at time: %s" % datetime.datetime.utcnow())




