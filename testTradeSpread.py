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
from readConf import *

bin_type = "_spread"
if bin_type == "_spread":
    bin_type = bin_type + str(spread -1)

fx = False
fx_position = 10000
fx_spread = 1

current_dir = os.path.dirname(__file__)

logging.config.fileConfig( os.path.join(current_dir,"config","logging.conf"))
logger = logging.getLogger("app")

closes_tmp = {}

def get_redis_data(db):
    print("DB_NO:", db_no)
    r = redis.Redis(host= host, port=6379, db=db_no, decode_responses=True)
    result = r.zrangebyscore(db, start_stp, end_stp, withscores=True)
    close_tmp, high_tmp, low_tmp = [], [], []
    time_tmp = []
    score_tmp = []
    spread_tmp = []
    payout_tmp = {}
    spread_dict = {}

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
        spr = tmps.get("spreadAus")/100
        spread_tmp.append(spr)

        if spr in spread_dict.keys():
            spread_dict[spr] = spread_dict[spr] + 1
        else:
            spread_dict[spr] = 1

        pay = tmps.get("payout")
        if pay in payout_tmp.keys():
            payout_tmp[pay] = payout_tmp[pay] + 1
        else:
            payout_tmp[pay] = 1

        if int(close_t) == 0:
            print("close:0 " + str(score))
        #high_tmp.append(tmps.get("high"))
        #low_tmp.append(tmps.get("low"))
        closes_tmp[score] = (close_t, tmps.get("spreadAus")/100)

    for i in spread_dict.keys():
        print("SPREAD:" + str(i), spread_dict[i])

    for i in payout_tmp.keys():
        print("PAYOUT:" + str(i), payout_tmp[i])

    close = 10000 * np.log(close_tmp/shift(close_tmp, 1, cval=np.NaN) )[1:]
    #high = 10000 * np.log(high_tmp / shift(high_tmp, 1, cval=np.NaN) )[1:]
    #low = 10000 * np.log(low_tmp / shift(low_tmp, 1, cval=np.NaN)  )[1:]

    close_data, high_data, low_data, time_data, price_data , predict_time_data, predict_score_data , end_price_data \
        = [], [], [], [], [], [], [], []
    label_data = []
    spread_data = []
    spread0 , spread1, spread2,spread3 , spread4, spread5, spread6over = 0,0,0,0,0,0,0

    data_length = len(close) - maxlen - pred_term -1
    print("data_length:" + str(data_length))

    up = 0
    same = 0

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
        if except_highlow:
            if datetime.strptime(time_tmp[1 + i + maxlen -1], '%Y-%m-%d %H:%M:%S').hour in except_list:
                continue;
        #maxlen前の時刻までつながっていないものは除外。たとえば日付またぎなど
        tmp_time_bef = datetime.strptime(time_tmp[1 + i], '%Y-%m-%d %H:%M:%S')
        tmp_time_aft = datetime.strptime(time_tmp[1 + i + maxlen -1], '%Y-%m-%d %H:%M:%S')
        delta =tmp_time_aft - tmp_time_bef

        if delta.total_seconds() > ((maxlen-1) * int(s)):
            #print(tmp_time_aft)
            continue;
        close_data.append(close[i:(i + maxlen)])
        time_data.append(time_tmp[1 + i + maxlen -1])
        price_data.append(close_tmp[1 + i + maxlen -1])
        spr = spread_tmp[1 + i + maxlen -1]
        spread_data.append(spr)
        if spr == 0.001:
            spread1 = spread1 +1
        elif spr == 0.002:
            spread2 = spread2 + 1
        elif spr == 0.003:
            spread3 = spread3 + 1
        elif spr == 0.004:
            spread4 = spread4 + 1
        elif spr == 0.005:
            spread5 = spread5 + 1
        elif spr >= 0.006:
            spread6over = spread6over + 1
        elif spr < 0.001:
            spread0 = spread0 + 1

        predict_time_data.append(time_tmp[1 + i + maxlen])
        predict_score_data.append(score_tmp[1 + i + maxlen ])
        end_price_data.append(close_tmp[1 + i + maxlen + pred_term - 1])

        #high_data.append(high[i:(i + maxlen)])
        #low_data.append(low[i:(i + maxlen)])

        bef = close_tmp[1 + i + maxlen -1]
        aft = close_tmp[1 + i + maxlen + pred_term -1]
        # 正解をいれる
        lbl,up_cnt,same_cnt = get_label_data(bef, aft, spr, up, same)
        up = up_cnt
        same = same_cnt
        label_data.append(lbl)

    close_np = np.array(close_data)
    time_np = np.array(time_data)
    price_np = np.array(price_data)

    predict_time_np = np.array(predict_time_data)
    predict_score_np = np.array(predict_score_data)
    end_price_np = np.array(end_price_data)

    close_tmp_np = np.array(close_tmp)
    time_tmp_np = np.array(time_tmp)
    spread_np = np.array(spread_data)
    #high_np = np.array(high_data)
    #low_np = np.array(low_data)

    retX = np.zeros((len(close_np), maxlen, in_num))
    retX[:, :, 0] = close_np[:]
    #retX[:, :, 1] = high_np[:]
    #retX[:, :, 2] = low_np[:]

    retY = np.array(label_data)
    #retZ = np.array(label_dataZ)

    print("X SHAPE:", retX.shape)
    print("Y SHAPE:", retY.shape)
    print("UP: ",up/len(retY))
    print("SAME: ", same / len(retY))
    print("DOWN: ", (len(retY) - up - same) / len(retY))
    spread_total = spread1 + spread2 + spread3 + spread4 + spread5 + spread6over + spread0

    print("spread total: ", spread_total)
    print("spread0: ", spread0 / spread_total)
    print("spread1: ", spread1 / spread_total)
    print("spread2: ", spread2 / spread_total)
    print("spread3: ", spread3 / spread_total)
    print("spread4: ", spread4 / spread_total)
    print("spread5: ", spread5 / spread_total)
    print("spread6over: ", spread6over / spread_total)

    return retX, retY, price_np, time_np, close_tmp_np,  predict_time_np, predict_score_np, end_price_np, spread_np

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

def do_predict(test_X):

    model = get_model()
    if model is None:
        return None

    res = model.predict(test_X, verbose=0, batch_size=batch_size)
    print("Predict finished")

    K.clear_session()
    return res

def get_label_data(bef, aft, spr, up, same):
    res = [0, 1, 0]
    if float(Decimal(str(aft)) - Decimal(str(bef))) >= float(Decimal("0.001") + Decimal(str(spr))):
        # 上がった場合
        res = [1, 0, 0]
        up = up +1
    elif float(Decimal(str(bef)) - Decimal(str(aft))) >= float(Decimal("0.001") + Decimal(str(spr))):
        res = [0, 0, 1]
    else:
        same = same +1
    return res, up, same

if __name__ == "__main__":
    predicts = {}
    trades = {}

    trade_cnt = 0
    trade_win_cnt = 0

    predict_cnt = 0
    predict_win_cnt = 0

    r = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)
    tradeReults = r.zrangebyscore(symbol + db_suffix + "_TRADE", start_stp, end_stp, withscores=True)

    #spread0, spread1, spread2, spread3, spread4, spread5,spread6over = 0, 0, 0, 0, 0, 0, 0
    spread0_trade, spread1_trade, spread2_trade, spread3_trade, spread4_trade, spread5_trade, spread6over_trade = 0, 0, 0, 0, 0, 0, 0
    spread0_trade_win, spread1_trade_win, spread2_trade_win, spread3_trade_win, spread4_trade_win\
        , spread5_trade_win, spread6over_trade_win = 0, 0, 0, 0, 0, 0, 0

    for tradeReult in tradeReults:
        body = tradeReult[0]
        score = tradeReult[1]
        tmps = json.loads(body)
        #startVal = tmps.get("startVal")
        #endVal = tmps.get("endVal")
        result = tmps.get("result")
        trades[score] = tmps.get("result")

    dataX, dataY, price_data, time_data, close, predict_time, predict_score, end_price, spread_data = get_redis_data(symbol + db_suffix )
    res = do_predict(dataX)

    ind5 = np.where(res >=border)[0]
    x5 = res[ind5,:]
    y5= dataY[ind5,:]
    #z5 = dataZ[ind5, :]
    p5 = price_data[ind5]
    t5 = time_data[ind5]
    pt5= predict_time[ind5]
    ps5 = predict_score[ind5]
    ep5 = end_price[ind5]
    sp5 = spread_data[ind5]

    for x,y,p,t,pt,ps,ep,sp in zip(x5,y5,p5,t5, pt5, ps5, ep5, sp5):

        if x.argmax() == 0 or x.argmax() == 2:
            predict_cnt = predict_cnt + 1
            if x.argmax() == y.argmax():
                predicts[ps] = ("win", sp)
                predict_win_cnt = predict_win_cnt + 1
            else:
                predicts[ps] = ("lose", sp)

    predict_money = default_money
    trade_money = default_money
    tmp_start = start
    predict_money_arr = []
    trade_money_arr = []
    close_arr = []
    spread_arr = []

    sorted_close = sorted(closes_tmp.items(), key=lambda x: x[0])
    for list in sorted_close:
        score = list[0]
        close_arr.append(list[1][0])
        spread_arr.append(list[1][1])
        #score = int(time.mktime(tmp_start.timetuple()))
        if score in predicts.keys():
            sp = predicts[score][1]
            if predicts[score][0] == "win":
                predict_money = predict_money + payout
            else:
                predict_money = predict_money - payoff
            if score in trades.keys():
                trade_cnt = trade_cnt + 1
                trade_res = "win"
                if trades[score] == "win":
                    trade_money = trade_money + payout
                    trade_win_cnt = trade_win_cnt + 1
                elif trades[score] == "lose":
                    trade_money = trade_money - payoff
                    trade_res = "lose"

                if sp == 0.001:
                    spread1_trade = spread1_trade + 1
                    if trade_res == "win":
                        spread1_trade_win = spread1_trade_win + 1
                elif sp == 0.002:
                    spread2_trade = spread2_trade + 1
                    if trade_res == "win":
                        spread2_trade_win = spread2_trade_win + 1
                elif sp == 0.003:
                    spread3_trade = spread3_trade + 1
                    if trade_res == "win":
                        spread3_trade_win = spread3_trade_win + 1
                elif sp == 0.004:
                    spread4_trade = spread4_trade + 1
                    if trade_res == "win":
                        spread4_trade_win = spread4_trade_win + 1
                elif sp == 0.005:
                    spread5_trade = spread5_trade + 1
                    if trade_res == "win":
                        spread5_trade_win = spread5_trade_win + 1
                elif sp >= 0.006:
                    spread6over_trade = spread6over_trade + 1
                    if trade_res == "win":
                        spread6over_trade_win = spread6over_trade_win + 1
                elif sp < 0.001:
                    spread0_trade = spread0_trade + 1
                    if trade_res == "win":
                        spread0_trade_win = spread0_trade_win + 1


        predict_money_arr.append(predict_money)
        trade_money_arr.append(trade_money)


    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " Now Plotting")
    fig = plt.figure()
    #価格の遷移
    ax1 = fig.add_subplot(2,1,1)
    #ax1.plot(time,close)
    ax1.plot(close_arr, 'g')

    ax2 = ax1.twinx()

    ax2.plot(trade_money_arr,"m")
    ax2.plot(predict_money_arr, "b")

    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(spread_arr, 'g')

    print("real trade cnt:", len(tradeReults))
    print("trade cnt: " + str(trade_cnt))
    print("spread0_trade cnt: ", spread0_trade )
    print("spread1_trade cnt: ", spread1_trade )
    print("spread2_trade cnt: ", spread2_trade )
    print("spread3_trade cnt: ", spread3_trade )
    print("spread4_trade cnt: ", spread4_trade )
    print("spread5_trade cnt: ", spread5_trade)
    print("spread6over_trade cnt: ", spread6over_trade)

    print("spread0_trade_win rate: ", 0 if spread0_trade == 0 else spread0_trade_win/spread0_trade )
    print("spread1_trade_win rate: ", 0 if spread1_trade == 0 else spread1_trade_win/spread1_trade )
    print("spread2_trade_win rate: ", 0 if spread2_trade == 0 else spread2_trade_win/spread2_trade )
    print("spread3_trade_win rate: ", 0 if spread3_trade == 0 else spread3_trade_win/spread3_trade )
    print("spread4_trade_win rate: ", 0 if spread4_trade == 0 else spread4_trade_win/spread4_trade )
    print("spread5_trade_win rate: ", 0 if spread5_trade == 0 else spread5_trade_win/spread5_trade )
    print("spread6over_trade_win rate: ", 0 if spread6over_trade == 0 else spread6over_trade_win/spread6over_trade )

    print("trade correct: " + str(trade_win_cnt / trade_cnt))
    print("trade money: " + str(trade_money))

    print("predict cnt : " + str(predict_cnt))
    print("predict correct : " + str(predict_win_cnt / predict_cnt))
    print("predict money: " + str(predict_money))

    print("trade cnt rate: " + str(trade_cnt / predict_cnt))
    plt.show()