
from datetime import datetime
import time
import redis

#symbol = "AUDUSD"
symbol = "GBPJPY"

#symbols = (symbol+"2", symbol+"5") #hybrid用
symbol_list = {"opt": (symbol + "_30_SPR_OPT",),
               "snc": (symbol + "_30_SPR_SNC",),
               "fil": (symbol + "_30_TRB",),
               "noriyasu": (symbol + "_30_TRB",symbol + "_30_SPR",symbol + "_60_SPR",),
               "ig": (symbol + "_2_IG",),
               }

import_db_nos = {"opt":8,"snc":8,"noriyasu":8,"yorioko":14,"fil":14,"ig":6}
export_db_no = 8

import_db = "snc"
export_host = "amd3"
import_host = "127.0.0.1"

demo_flg = False
db_post_fix = ""
if demo_flg:
    db_post_fix = "_DEMO"


start = datetime(2020, 2, 2)
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2020, 6, 2)
end_stp = int(time.mktime(end.timetuple()))

def import_data():
    import_db_no = import_db_nos.get(import_db)
    symbols = symbol_list.get(import_db)

    export_r = redis.Redis(host= export_host, port=6379, db=export_db_no)
    import_r = redis.Redis(host= import_host, port=6379, db=import_db_no)

    for sym in symbols:
        db = sym + db_post_fix
        result_data = export_r.zrangebyscore(db, start_stp, end_stp, withscores=True)
        print(db)
        for line in result_data:
            body = line[0]
            score = line[1]
            imp = import_r.zrangebyscore(db , score, score)
            if len(imp) == 0:
                import_r.zadd(db, body, score)

        trade_db = sym + "_TRADE" + db_post_fix
        print(trade_db)
        result_trade_data = export_r.zrangebyscore(trade_db, start_stp, end_stp, withscores=True)

        for line in result_trade_data:
            body = line[0]
            score = line[1]
            imp = import_r.zrangebyscore(trade_db, score, score)
            if len(imp) == 0:
                import_r.zadd(trade_db, body, score)

        import_r.save()


if __name__ == "__main__":
    import_data()
