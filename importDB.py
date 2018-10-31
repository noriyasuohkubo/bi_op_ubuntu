
import redis
from datetime import datetime
import time


#symbol = "AUDUSD"
symbol = "GBPJPY"
#symbol = "EURUSD"

#symbols = (symbol+"2", symbol+"5") #hybrid用
symbol_list = {"ubuntu1":(symbol, ),"ubuntu2":(symbol+"1",symbol+"2",symbol+"3",symbol+"4", symbol+"5"),"ubuntu3":(symbol+"_OPT"+"1",symbol+"_OPT"+"2",symbol+"_OPT"+"3",symbol+"_OPT"+"4", symbol+"_OPT"+"5")}
import_db_nos = {"ubuntu1":11,"ubuntu2":12,"ubuntu3":13,}
db_suffix_trade_list = {"ubuntu1":"","ubuntu2":"","ubuntu3":"_OPT",}
export_db_no = 8

export_host = "ubuntu1"
import_host = "127.0.0.1"


start = datetime(2018, 10, 29)
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2018, 11, 1)
end_stp = int(time.mktime(end.timetuple()))

def import_data():
    import_db_no = import_db_nos.get(export_host)
    symbols = symbol_list.get(export_host)

    export_r = redis.Redis(host= export_host, port=6379, db=export_db_no)
    import_r = redis.Redis(host= import_host, port=6379, db=import_db_no)
    for sym in symbols:
        result_data = export_r.zrangebyscore(sym, start_stp, end_stp, withscores=True)

        for line in result_data:
            body = line[0]
            score = line[1]
            imp = import_r.zrangebyscore(sym , score, score)
            if len(imp) == 0:
                import_r.zadd(sym, body, score)
    db_suffix_trade = db_suffix_trade_list.get(export_host)
    result_trade_data = export_r.zrangebyscore(symbol + db_suffix_trade + "_TRADE", start_stp, end_stp, withscores=True)

    for line in result_trade_data:
        body = line[0]
        score = line[1]
        imp = import_r.zrangebyscore(symbol + db_suffix_trade + "_TRADE" , score, score)
        if len(imp) == 0:
            import_r.zadd(symbol + db_suffix_trade + "_TRADE", body, score)


if __name__ == "__main__":
    import_data()
