from pymongo import MongoClient
import datetime

# Accepts a flag ("Array", "MP_Array", "Dictionary", or "Queue") that determines the type of data structure to be inserted
# Opens a connection to the MongoDB at specified ip & port
# Authenticates with the username, password, & src arguments
# Inserts data into database-collection specified by database & collection arguments
# Returns a success boolean
def dump(flag, ip, port, username, password, src, database, collection, data):
    try:
        client = MongoClient(ip, port)
        client.admin.authenticate(username, password, mechanism='SCRAM-SHA-1', source=src)
        db = client[database]
        coll = db[collection]
        if (flag == "Array"):
            coll.insert_many(data)
            return True
        elif (flag == "Queue"):
            while not data.empty():
                coll.insert_one(data.get())
            return True
        elif (flag == "Dictionary"):
            coll.insert_one(data)
            return True
        elif (flag == "MP_Array"):
            for multiprocessing_object in data:
                element = multiprocessing_object.get()
                if element:
                    coll.insert_one(element)
            return True
        else:
            print("DATABASE DUMP ERROR: Unknown data structure. Check database.py for details")
            return False
    except Exception as error:
        print("DATABASE DUMP ERROR: Unable to insert record. Exception: ", error, " At time: ", datetime.datetime.utcnow())
        return False
        


# Creates UPS and TempSensor temporary collections with averaged
# values grouped by timestamp truncated to minutes
def temp_coll_generator(ip, port, username, password, src, database, collection):
    try:
        client = MongoClient(ip, port)
        client.admin.authenticate(username, password, mechanism='SCRAM-SHA-1', source=src)
        db = client[database]
        coll = db[collection]
        if (collection == "TempSensor"):
            pipeline = [
                {
                    u"$group": {
                        u"_id": {
                            u"$dateToString": {
                                u"format": u"%Y-%m-%dT%H:%M",
                                u"date": u"$Timestamp"
                            }
                        },
                        u"AVG15_M": {
                            u"$avg": u"$15_M"
                        },
                        u"AVG15_T": {
                            u"$avg": u"$15_T"
                        },
                        u"AVG16_M": {
                            u"$avg": u"$16_M"
                        },
                        u"AVG16_T": {
                            u"$avg": u"$16_T"
                        },
                        u"AVG18_M": {
                            u"$avg": u"$18_M"
                        },
                        u"AVG18_T": {
                            u"$avg": u"$18_T"
                        },
                        u"AVG22_M": {
                            u"$avg": u"$22_M"
                        },
                        u"AVG22_T": {
                            u"$avg": u"$22_T"
                        },
                        u"AVGCRAC1_IN": {
                            u"$avg": u"$CRAC1_IN"
                        },
                        u"AVGCRAC1_OUT": {
                            u"$avg": u"$CRAC1_OUT"
                        },
                        u"AVGCRAC2_IN": {
                            u"$avg": u"$CRAC2_IN"
                        },
                        u"AVGCRAC2_OUT": {
                            u"$avg": u"$CRAC2_OUT"
                        },
                        u"AVGCW_PIPE": {
                            u"$avg": u"$CW_PIPE"
                        },
                        u"AVGAH_8_OUT": {
                            u"$avg": u"$AH_8_OUT"
                        },
                        u"AVG30_M": {
                            u"$avg": u"$30_M"
                        },
                        u"AVG30_T": {
                            u"$avg": u"$30_T"
                        },
                        u"AVG28_M": {
                            u"$avg": u"$28_M"
                        },
                        u"AVG28_T": {
                            u"$avg": u"$28_T"
                        },
                        u"AVG26_M": {
                            u"$avg": u"$26_M"
                        },
                        u"AVG26_T": {
                            u"$avg": u"$26_T"
                        },
                        u"AVG24_M": {
                            u"$avg": u"$24_M"
                        },
                        u"AVG24_T": {
                            u"$avg": u"$24_T"
                        },
                        u"AVG14_M": {
                            u"$avg": u"$14_M"
                        },
                        u"AVG14_T": {
                            u"$avg": u"$14_T"
                        },
                        u"AVG12_M": {
                            u"$avg": u"$12_M"
                        },
                        u"AVG12_T": {
                            u"$avg": u"$12_T"
                        }
                    }
                },
                {
                    u"$out": u"TempSensor_Temp"
                }
            ]
        elif (collection == "UPS"):   
            pipeline = [
            {
                u"$group": {
                    u"_id": {
                        u"$dateToString": {
                            u"format": u"%Y-%m-%dT%H:%M",
                            u"date": u"$Timestamp"
                        }
                    },
                    u"AVGupsSecondsOnBattery": {
                        u"$avg": u"$upsSecondsOnBattery"
                    },
                    u"AVGupsOutputPercentLoad-1": {
                        u"$avg": u"$upsOutputPercentLoad-1"
                    },
                    u"AVGupsOutputPercentLoad-2": {
                        u"$avg": u"$upsOutputPercentLoad-2"
                    },
                    u"AVGupsOutputPercentLoad-3": {
                        u"$avg": u"$upsOutputPercentLoad-3"
                    }
                }
            }, 
            {
                u"$out": u"UPS_Temp"
            }
            ]
        elif (collection == "PDU"):
            pipeline = [
    {
        u"$match": {
            u"Rack Number": 19.0
        }
    }, 
    {
        u"$group": {
            u"_id": {
                u"$dateToString": {
                    u"format": u"%Y-%m-%dT%H:%M",
                    u"date": u"$Timestamp"
                }
            },
            u"TempF-1": {
                u"$avg": u"$rPDU2SensorTempHumidityStatusTempF-1"
            },
            u"TempF-2": {
                u"$avg": u"$rPDU2SensorTempHumidityStatusTempF-2"
            },
            u"TempF-3": {
                u"$avg": u"$rPDU2SensorTempHumidityStatusTempF-3"
            },
            u"TempF-4": {
                u"$avg": u"$rPDU2SensorTempHumidityStatusTempF-4"
            },
            u"RelativeHumidity-1": {
                u"$avg": u"$rPDU2SensorTempHumidityStatusRelativeHumidity-1"
            },
            u"RelativeHumidity-2": {
                u"$avg": u"$rPDU2SensorTempHumidityStatusRelativeHumidity-2"
            },
            u"RelativeHumidity-3": {
                u"$avg": u"$rPDU2SensorTempHumidityStatusRelativeHumidity-3"
            },
            u"RelativeHumidity-4": {
                u"$avg": u"$rPDU2SensorTempHumidityStatusRelativeHumidity-4"
            }
        }
    }, 
    {
        u"$out": u"PDU_Temp"
    }
]
            
        cursor = coll.aggregate(
            pipeline, 
            allowDiskUse = True
        )
        client.close()
        return True
    except Exception as e:
        print("DATABASE ERROR: Failed to construct temporary collections with error: ", e, "at time: ", datetime.datetime.utcnow())
        client.close()
        False

# Opens a connection to the MongoDB at the specified ip & port
# Authenticates with username, password, and src arguments
# Groups records by timestamp (truncated to minutes), averages fields for duplicate timestamps
# Returns cursor (collection) of records 
def extract(ip, port, username, password, src, database, collection):
    try:
        client = MongoClient(ip, port)
        client.admin.authenticate(username, password, mechanism='SCRAM-SHA-1', source=src)
        db = client[database]
        coll = db[collection]
        pipeline = [
                {
                    u"$match": {
                        u"Rack Number": {
                            u"$ne": 19.0
                        }
                    }
                }, 
                {
                    u"$group": {
                        u"_id": {
                            u"rackNo": u"$Rack Number",
                            u"date": {
                                u"$dateToString": {
                                    u"format": u"%Y-%m-%dT%H:%M",
                                    u"date": u"$Timestamp"
                                }
                            }
                        },
                        u"TempF-1": {
                            u"$avg": u"$rPDU2SensorTempHumidityStatusTempF-1"
                        },
                        u"TempF-2": {
                            u"$avg": u"$rPDU2SensorTempHumidityStatusTempF-2"
                        },
                        u"RelativeHumidity-1": {
                            u"$avg": u"$rPDU2SensorTempHumidityStatusRelativeHumidity-1"
                        },
                        u"RelativeHumidity-2": {
                            u"$avg": u"$rPDU2SensorTempHumidityStatusRelativeHumidity-2"
                        }
		    }
                }, 
                {
                    u"$group": {
                        u"_id": u"$_id.date",
                        u"Racks": {
                            u"$push": {
                                u"rackNo": u"$_id.rackNo",
                                u"TempF-1": u"$TempF-1",
                                u"TempF-2": u"$TempF-2",
                                u"RelativeHumidity-1": u"$RelativeHumidity-1",
                                u"RelativeHumidity-2": u"$RelativeHumidity-2",
                                u"UPS": u"$UPS",
                                u"TempSensor": u"$TempSensor",
                                u"Rack19": u"$Rack19"
                            }
                        }
                    }
                }, 
                {
                    u"$lookup": {
                        u"from": u"UPS_Temp",
                        u"localField": u"_id",
                        u"foreignField": u"_id",
                        u"as": u"UPS"
                    }
                }, 
                {
                    u"$lookup": {
                        u"from": u"TempSensor_Temp",
                        u"localField": u"_id",
                        u"foreignField": u"_id",
                        u"as": u"TempSensor"
                    }
                },
                {
        u"$lookup": {
            u"from": u"PDU_Temp",
            u"localField": u"_id",
            u"foreignField": u"_id",
            u"as": u"Rack19"
        }
    }
            ]

 
        cursor = coll.aggregate(
            pipeline, 
            allowDiskUse = True
        )
            
        try:
            return cursor
            client.close()
        except:
            raise Exception("Unable to return cursor.")
    except Exception as error:
        print("DATABASE EXTRACT ERROR: Unable to retrieve records. Exception: ", error, " At time: ", datetime.datetime.utcnow())
        client.close()
        return {}




