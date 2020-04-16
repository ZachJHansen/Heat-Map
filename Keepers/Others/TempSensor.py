from pymongo import MongoClient
from threading import Thread
import database
import subprocess
import datetime
import sys

# Path to probeConfig.txt, which determines naming for cURL response
probeConfig = sys.argv[1]

# Instances of the TempSensor class will have the following fields and methods:

# ip: IP address 
# download_success: true if the curl request to the specified IP is successful
# parse_success: true if the response returned by curl can be parsed successfully
# record: a dictionary to be inserted into the MongoDB

# querySensor: Attempts to fetch temperature data via curl from IP
# parseResponse: Turns the response string into a collection of key-value pairs (self.record)

class TempSensor():
    def __init__(self, ip):
        self.ip = ip
        self.download_success = False
        self.parse_success = False
        self.record = {}
        self.key_list = []

    def querySensor(self):
        try:
            self.response = subprocess.check_output("curl -s http://" + self.ip + "/temp", shell=True, stderr=subprocess.STDOUT).decode('ascii')
            self.download_success = True
        except:
            print("TEMPSENSOR ERROR: Curl request to IP ", self.ip, " failed at: ", datetime.datetime.utcnow())

    def parseProbeConfig(self):
        with open(probeConfig, "r") as probe:
            lines = probe.readlines()
            for line in lines:
                if (line != "\n"):
                    split_resp = line.split('\t')
                    name = split_resp[2]
                    name.strip()
                    self.key_list.append(name)
            # Cut out last 20 elements (Quanah probes)
            self.key_list = self.key_list[0:len(self.key_list)-20]

    def parseResponse(self):
        try:
            self.record["IP Address"] = self.ip
            self.record["Timestamp"] = datetime.datetime.utcnow()
            split_resp = self.response.split("|")
            N = len(split_resp)
            i = 0
            # IP 10.1.12.1 stores data from AH_8_OUT to 12_T, IP 10.1.18.1 stores data from CRAC1_IN to CW_PIPE_COOLING
            if (self.ip == "10.1.12.1"):
                j = 1 
            else:
                j = 17
            while (i < N):
                key = self.key_list[j]
                value = split_resp[i+1]
                contains = key.find("MISSING")
                if (contains == -1):
                    self.record[key] = float(value)
                i += 2
                j += 1
            self.parse_success = True
        except Exception as error:
            print("TEMPSENSOR ERROR: Encountered error: {", error, "}. While parsing curl response at time ", datetime.datetime.utcnow())
    
# End TempSensor Class

# Creates an instance of TempSensor class for a given IP address
# Returns a tuple; (success_boolean, dictionary) 
def initTempSensorQuery(ip):
    sensor = TempSensor(ip)
    sensor.querySensor()
    if sensor.download_success:
        sensor.parseProbeConfig()
        sensor.parseResponse()
        if sensor.parse_success:
            return (True, sensor.record)
    return (False, {})          

# Main
if __name__=="__main__":
     results = []

     result1 = initTempSensorQuery("10.1.18.1") 
     if result1[0]:
         results.append(result1[1])
     else:
         print("TEMPSENSOR ERROR: Request to IP 10.1.18.1 failed to query or parse successfully")

     result2 = initTempSensorQuery("10.1.12.1")
     if result2[0]:
         results.append(result2[1])
     else:
         print("TEMPSENSOR ERROR: Request to IP 10.1.12.1 failed to query or parse successfully")

     success = database.dump('Array', '10.1.6.102', 27017, 'logger', '3mpZjGdS', 'monitoring', 'test', 'TempSensor', results)
     if not success:
         print("TEMPSENSOR ERROR: See traceback above")






                
