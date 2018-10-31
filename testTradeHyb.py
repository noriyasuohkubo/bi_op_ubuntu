import numpy as np
import keras.models
import tensorflow as tf
import configparser
import os
import redis
import traceback
import json
from scipy.ndimage.interpolation import shift
import logging.config
from keras.models import load_model
from keras import backend as K
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime
from datetime import timedelta
from keras.utils.training_utils import multi_gpu_model
import time
from indices import index
from decimal import Decimal

#symbol = "AUDUSD"
symbol = "GBPJPY"
#symbol = "EURUSD"

gpu_count = 1
maxlen = 300
pred_term = 6
rec_num = 10000 + maxlen + pred_term + 1
batch_size = 8192 * gpu_count

start = datetime(2018, 7, 30,22)
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2018, 9, 1 )
end_stp = int(time.mktime(end.timetuple()))

host = "127.0.0.1"
db_no = 12
s = "5"
int_s = int(s)

#db_suffixs = (1,2,3,4,5)
db_suffixs = (5,)
db_suffix_trade = ""


#suffix = ".40*30"
suffix = ".28*15"
askbid = "_bid"

except_index = False
except_highlow = True

drop = 0.1
np.random.seed(0)
n_hidden =  30
n_hidden2 = 0
n_hidden3 = 0
n_hidden4 = 0

border = 0.56
payout = 950
default_money = 1005000

spread = 1

fx = False
fx_position = 10000
fx_spread = 1

in_num=1

current_dir = os.path.dirname(__file__)
ini_file = os.path.join(current_dir,"config","config.ini")
config = configparser.ConfigParser()
config.read(ini_file)
MODEL_DIR = config['lstm']['MODEL_DIR']

type = 'bydrop'

file_prefix = symbol + "_bydrop_in" + str(in_num) + "_" + s + "_m" + str(maxlen) + "_term_" + str(pred_term * int(s)) + "_hid1_" + str(n_hidden) + \
                          "_hid2_" + str(n_hidden2) + "_hid3_" + str(n_hidden3) + "_hid4_" + str(n_hidden4) + "_drop_" + str(drop) + askbid

logging.config.fileConfig( os.path.join(current_dir,"config","logging.conf"))
logger = logging.getLogger("app")

model_file = os.path.join(MODEL_DIR, file_prefix +".hdf5" + suffix)

closes_tmp = {}

def get_redis_data(db):
    print("DB_NO:", db_no)
    r = redis.Redis(host= host, port=6379, db=db_no, decode_responses=True)
    result = r.zrangebyscore(db, start_stp, end_stp, withscores=True)
    #result = r.zrevrange(symbol, 0  , rec_num  , withscores=False)
    close_tmp, high_tmp, low_tmp = [], [], []
    time_tmp = []
    score_tmp = []
    print(result[0:5])
    print(result[-5:])
    #result.reverse()
    #print(index)
    indicies = np.ones(len(index), dtype=np.int32)
    #経済指標発表前後2時間は予想対象からはずす
    for i,ind in enumerate(index):
        tmp_datetime = datetime.strptime(ind, '%Y-%m-%d %H:%M:%S')
        indicies[i] = int(time.mktime(tmp_datetime.timetuple()))

    for line in result:
        body = line[0]
        score = line[1]
        tmps = json.loads(body)
        close_t = tmps.get("close")
        close_tmp.append(tmps.get("close"))
        time_tmp.append(tmps.get("time"))
        score_tmp.append(score)
        if close_t == 0.0:
            print("close:0 " + str(score))
        #high_tmp.append(tmps.get("high"))
        #low_tmp.append(tmps.get("low"))
        closes_tmp[score] = close_t

    close = 10000 * np.log(close_tmp/shift(close_tmp, 1, cval=np.NaN) )[1:]
    #high = 10000 * np.log(high_tmp / shift(high_tmp, 1, cval=np.NaN) )[1:]
    #low = 10000 * np.log(low_tmp / shift(low_tmp, 1, cval=np.NaN)  )[1:]

    close_data, high_data, low_data, label_data, time_data, price_data , predict_time_data, predict_score_data , end_price_data \
        = [], [], [], [], [], [], [], [], []

    up =0
    same =0
    data_length = len(close) - maxlen - pred_term -1
    print("data_length:" + str(data_length))

    for i in range(data_length):
        continue_flg = False

        if except_index:
            tmp_datetime = datetime.strptime(time_tmp[1 + i + maxlen -1], '%Y-%m-%d %H:%M:%S')
            score = int(time.mktime(tmp_datetime.timetuple()))
            for ind in indicies:
                ind_datetime = datetime.fromtimestamp(ind)

                bef_datetime = ind_datetime - timedelta(hours=1)
                aft_datetime = ind_datetime + timedelta(hours=1)
                bef_time = int(time.mktime(bef_datetime.timetuple()))
                aft_time = int(time.mktime(aft_datetime.timetuple()))

                if bef_time <= score and score <= aft_time:
                    continue_flg = True
                    break;

            if continue_flg:
                continue;
        #ハイローオーストラリアの取引時間外を学習対象からはずす
        except_list = [20, 21, 22]
        if except_highlow:
            if datetime.strptime(time_tmp[1 + i + maxlen -1], '%Y-%m-%d %H:%M:%S').hour in except_list:
                continue;
        #maxlen前の時刻までつながっていないものは除外。たとえば日付またぎなど
        tmp_time_bef = datetime.strptime(time_tmp[1 + i], '%Y-%m-%d %H:%M:%S')
        tmp_time_aft = datetime.strptime(time_tmp[1 + i + maxlen -1], '%Y-%m-%d %H:%M:%S')
        delta =tmp_time_aft - tmp_time_bef

        if delta.total_seconds() > ((maxlen-1) * int_s):
            #print(tmp_time_aft)
            continue;
        close_data.append(close[i:(i + maxlen)])
        time_data.append(time_tmp[1 + i + maxlen -1])
        price_data.append(close_tmp[1 + i + maxlen -1])

        predict_time_data.append(time_tmp[1 + i + maxlen])
        predict_score_data.append(score_tmp[1 + i + maxlen ])
        end_price_data.append(close_tmp[1 + i + maxlen + pred_term - 1])

        #high_data.append(high[i:(i + maxlen)])
        #low_data.append(low[i:(i + maxlen)])

        bef = close_tmp[1 + i + maxlen -1]
        aft = close_tmp[1 + i + maxlen + pred_term -1]

        #正解をいれる
        if float(Decimal(str(aft)) - Decimal(str(bef))) >= 0.001 * spread:
            #上がった場合
            label_data.append([1,0,0])
            up = up + 1
        elif  float(Decimal(str(bef)) - Decimal(str(aft))) >= 0.001 * spread:
            label_data.append([0,0,1])
        else:
            label_data.append([0,1,0])
            same = same + 1

    close_np = np.array(close_data)
    time_np = np.array(time_data)
    price_np = np.array(price_data)

    predict_time_np = np.array(predict_time_data)
    predict_score_np = np.array(predict_score_data)
    end_price_np = np.array(end_price_data)

    close_tmp_np = np.array(close_tmp)
    time_tmp_np = np.array(time_tmp)

    #high_np = np.array(high_data)
    #low_np = np.array(low_data)

    retX = np.zeros((len(close_np), maxlen, in_num))
    retX[:, :, 0] = close_np[:]
    #retX[:, :, 1] = high_np[:]
    #retX[:, :, 2] = low_np[:]

    retY = np.array(label_data)

    print("X SHAPE:", retX.shape)
    print("Y SHAPE:", retY.shape)
    print("UP: ",up/len(retY))
    print("SAME: ", same / len(retY))
    print("DOWN: ", (len(retY) - up - same) / len(retY))

    return retX, retY, price_np, time_np, close_tmp_np,  predict_time_np, predict_score_np, end_price_np

def get_model():

    if os.path.isfile(model_file):
        model = load_model(model_file)
        #model_gpu = multi_gpu_model(model, gpus=gpu_count)
        model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
        print("Load Model")
        return model
    else:
        print("the Model not exists!")
        return None

def do_predict(test_X, test_Y):

    model = get_model()
    if model is None:
        return None

    total_num = len(test_Y)
    res = model.predict(test_X, verbose=0, batch_size=batch_size)
    print("Predict finished")

    K.clear_session()
    return res


if __name__ == "__main__":
    predicts = {}
    trades = {}

    trade_cnt = 0
    trade_win_cnt = 0

    predict_cnt = 0
    predict_win_cnt = 0

    predict_cnts = [0,0,0,0,0]
    predict_win_cnts = [0, 0, 0, 0, 0]

    r = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)
    tradeReults = r.zrangebyscore(symbol + "_TRADE" + db_suffix_trade, start_stp, end_stp, withscores=True)
    for tradeReult in tradeReults:
        body = tradeReult[0]
        score = tradeReult[1]

        tmps = json.loads(body)
        #startVal = tmps.get("startVal")
        #endVal = tmps.get("endVal")
        result = tmps.get("result")
        trades[score] = tmps.get("result")

    for db_index in db_suffixs:

        dataX, dataY, price_data, time_data, close, predict_time, predict_score, end_price = get_redis_data(symbol + str(db_index))
        res = do_predict(dataX,dataY)

        ind5 = np.where(res >=border)[0]
        x5 = res[ind5,:]
        y5= dataY[ind5,:]
        p5 = price_data[ind5]
        t5 = time_data[ind5]
        pt5= predict_time[ind5]
        ps5 = predict_score[ind5]
        ep5 = end_price[ind5]

        #print(t5[0:10])

        up = res[:, 0]
        down = res[:, 2]
        up_ind5 = np.where(up >= border)[0]
        down_ind5 = np.where(down >= border)[0]

        x5_up = res[up_ind5,:]
        y5_up= dataY[up_ind5,:]
        p5_up = price_data[up_ind5]
        t5_up = time_data[up_ind5]
        up_total_length = len(x5_up)
        up_eq = np.equal(x5_up.argmax(axis=1), y5_up.argmax(axis=1))
        up_cor_length = int(len(np.where(up_eq == True)[0]))
        up_wrong_length = int(up_total_length - up_cor_length)

        x5_down = res[down_ind5,:]
        y5_down= dataY[down_ind5,:]
        p5_down = price_data[down_ind5]
        t5_down = time_data[down_ind5]
        down_total_length = len(x5_down)
        down_eq = np.equal(x5_down.argmax(axis=1), y5_down.argmax(axis=1))
        down_cor_length = int(len(np.where(down_eq == True)[0]))
        down_wrong_length = int(down_total_length - down_cor_length)

        predict_cnts[db_index -1] = up_total_length + down_total_length
        predict_win_cnts[db_index - 1] = up_cor_length + down_cor_length

        for x,y,p,t,pt,ps,ep in zip(x5,y5,p5,t5, pt5, ps5, ep5):

            if x.argmax() == 0 or x.argmax() == 2:
                predict_cnt = predict_cnt +1
                if x.argmax() == y.argmax():
                    predicts[ps] = "win"
                    predict_win_cnt = predict_win_cnt +1
                else:
                    predicts[ps] = "lose"


    predict_money = default_money
    trade_money = default_money
    tmp_start = start
    predict_money_arr = []
    trade_money_arr = []
    close_arr = []

    sorted_close = sorted(closes_tmp.items(), key=lambda x: x[0])
    for list in sorted_close:
        score = list[0]
        close_arr.append(list[1])

        #score = int(time.mktime(tmp_start.timetuple()))
        if score in predicts.keys():
            if predicts[score] == "win":
                predict_money = predict_money + payout
            else:
                predict_money = predict_money - 1000
            if score in trades.keys():
                trade_cnt = trade_cnt + 1
                if trades[score] == "win":
                    trade_money = trade_money + payout
                    trade_win_cnt = trade_win_cnt + 1
                else:
                    trade_money = trade_money - 1000

        predict_money_arr.append(predict_money)
        trade_money_arr.append(trade_money)


    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " Now Plotting")
    fig = plt.figure()
    #価格の遷移
    ax1 = fig.add_subplot(111)
    #ax1.plot(time,close)
    ax1.plot(close_arr, 'g')

    ax2 = ax1.twinx()

    ax2.plot(trade_money_arr,"m")
    ax2.plot(predict_money_arr, "b")

    print("trade cnt: " + str(trade_cnt))
    if trade_cnt != 0:
        print("trade correct: " + str(trade_win_cnt / trade_cnt))
    print("trade money: " + str(trade_money))

    for db_index in db_suffixs:
        print("predict cnt " + str(db_index) + ": " + str(predict_cnts[db_index -1]))
        if predict_cnts[db_index -1] != 0:
            print("predict correct " + str(db_index) + ": "  + str(predict_win_cnts[db_index -1] / predict_cnts[db_index -1]))

    print("predict cnt all: " + str(predict_cnt))
    if predict_cnt!= 0:
        print("predict correct all: " + str(predict_win_cnt / predict_cnt))
    print("predict money: " + str(predict_money))
    plt.show()