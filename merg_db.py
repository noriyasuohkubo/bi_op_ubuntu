
from datetime import datetime
import time
import redis



"""
DBデータをマージする
"""

dbs = ["GBPJPY_30_SPR", "GBPJPY_30_SPR_TRADE"]

org_db_host = "localhost"
org_db_no = 8

imp_db_host = "amd6"
imp_db_no = 8


start = datetime(2020, 12, 17, 23)
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2020, 12, 18, 23)
end_stp = int(time.mktime(end.timetuple()))

def import_data():
    imp_r = redis.Redis(host=imp_db_host, port=6379, db=imp_db_no)
    org_r = redis.Redis(host=org_db_host, port=6379, db=org_db_no)

    for db in dbs:

        result_data = imp_r.zrangebyscore(db, start_stp, end_stp, withscores=True)

        print(db)

        for line in result_data:
            body = line[0]
            score = line[1]

            org = org_r.zrangebyscore(db , score, score)
            #データなければ登録
            if len(org) == 0:
                org_r.zadd(db, body, score)

    org_r.save()


if __name__ == "__main__":
    import_data()
