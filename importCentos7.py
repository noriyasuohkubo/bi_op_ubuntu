
from datetime import datetime
import time
import redis

symbol = "GBPJPY"


start = datetime(2019, 1, 1)
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2019, 2, 1)
end_stp = int(time.mktime(end.timetuple()))

def import_data():

    export_r = redis.Redis(host= "centos7", port=6379, db=3)
    import_r = redis.Redis(host= "localhost", port=6379, db=0)

    db = symbol
    result_data = export_r.zrangebyscore(db, start_stp, end_stp, withscores=True)
    for line in result_data:
        body = line[0]
        score = line[1]
        imp = import_r.zrangebyscore(db , score, score)
        if len(imp) == 0:
            import_r.zadd(db, body, score)

    import_r.save()


if __name__ == "__main__":
    import_data()
