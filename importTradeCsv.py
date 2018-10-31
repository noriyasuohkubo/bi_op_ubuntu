import csv
import redis
from datetime import datetime
import time
import json

symbol = "GBPJPY_TRADE"

file_path = "/tmp/HL100005_2018_10_02_02_34_18.csv"
#file_path = ""
csv_file = open(file_path, "r", encoding="shift-jis", errors="", newline="" )
csv_reader = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\n", quotechar='"', skipinitialspace=True)

r = redis.Redis(host= "localhost", port=6379, db=11)

for row in csv_reader:
    #print(row)
    body = {"startVal":row[4]}
    body["endVal"] = row[8]
    body["time"] = row[9].split(" ")[1]
    body["result"] = "win"
    if row[7] == "---":
        body["result"] = "lose"
    body["timeStr"] = row[9].split(" ")[0].replace("/", "-") + "T" + row[9].split(" ")[1] + ".000Z"
    score = int(time.mktime(datetime.strptime(row[9].replace("/", "-") , '%Y-%m-%d %H:%M:%S').timetuple()))

    imp = r.zrangebyscore(symbol, score, score)
    if len(imp) == 0:
        #print(body)
        r.zadd(symbol , json.dumps(body), score)
"""
tradeReult = r.zrange(symbol  , 0, 10)
tmps = json.loads(tradeReult[0].decode('utf-8'))
startVal = tmps.get("startVal")
print(startVal)
"""