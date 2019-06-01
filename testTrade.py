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

merg = ""
if merg != "":
    merg = "_merg_" + merg

fx = False
fx_position = 10000
fx_spread = 1

current_dir = os.path.dirname(__file__)

logging.config.fileConfig( os.path.join(current_dir,"config","logging.conf"))
logger = logging.getLogger("app")

percentUP_cnt ={}

def get_redis_data():
    print("DB_NO:", db_no)
    r = redis.Redis(host= host, port=6379, db=db_no, decode_responses=True)
    result = r.zrangebyscore(symbol + db_suffix, start_stp, end_stp, withscores=True)
    #result = r.zrevrange(symbol, 0  , rec_num  , withscores=False)
    close_tmp, high_tmp, low_tmp = [], [], []
    time_tmp = []
    score_tmp = []
    spread_tmp = []
    payout_tmp = {}

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
        #print("spread",tmps.get("spread"))
        if tmps.get("spread") == None:
            spread_tmp.append(0.0)
        else:
            spread_tmp.append(tmps.get("spread"))

        if tmps.get("payout") == None:
            pay = 0.000
        else:
            pay = tmps.get("payout")

        if pay in payout_tmp.keys():
            payout_tmp[pay] = payout_tmp[pay] + 1
        else:
            payout_tmp[pay] = 1

        if close_t == 0.0:
            print("close:0 " + str(score))
        #high_tmp.append(tmps.get("high"))
        #low_tmp.append(tmps.get("low"))
    #Payoutの種類確認
    for i in payout_tmp.keys():
        print("PAYOUT:" + str(i), payout_tmp[i])

    close = 10000 * np.log(close_tmp/shift(close_tmp, 1, cval=np.NaN) )[1:]
    #high = 10000 * np.log(high_tmp / shift(high_tmp, 1, cval=np.NaN) )[1:]
    #low = 10000 * np.log(low_tmp / shift(low_tmp, 1, cval=np.NaN)  )[1:]

    close_data, high_data, low_data, label_data, time_data, price_data , predict_time_data, predict_score_data , end_price_data \
        = [], [], [], [], [], [], [], [], []
    spread_data = []
    spread_cnt = {}

    up =0
    same =0
    data_length = len(close) - maxlen - pred_term -1
    print("data_length:" + str(data_length))
    print(close[0:5])
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

        spr = spread_tmp[1 + i + maxlen - 1]
        if limit_border_flg:
            if spr <= border_spread:
                continue

        close_data.append(close[i:(i + maxlen)])
        time_data.append(time_tmp[1 + i + maxlen -1])
        price_data.append(close_tmp[1 + i + maxlen -1])

        predict_time_data.append(time_tmp[1 + i + maxlen])
        predict_score_data.append(score_tmp[1 + i + maxlen ])
        end_price_data.append(close_tmp[1 + i + maxlen + pred_term - 1])

        spread_data.append(spr)

        flg = False
        for k, v in spread_list.items():
            if spr > v[0] and spr <= v[1]:
                spread_cnt[k] = spread_cnt.get(k,0) + 1
                flg = True
                break
        if flg == False:
            if spr < 0:
                spread_cnt["spread0"] = spread_cnt.get("spread0", 0) + 1
            else:
                spread_cnt["spread16Over"] = spread_cnt.get("spread16Over",0) + 1

        #high_data.append(high[i:(i + maxlen)])
        #low_data.append(low[i:(i + maxlen)])

        bef = close_tmp[1 + i + maxlen -1]
        aft = close_tmp[1 + i + maxlen + pred_term -1]

        #正解をいれる
        if float(Decimal(str(aft)) - Decimal(str(bef))) >= float(Decimal("0.001") * Decimal(str(spread))):
            #上がった場合
            label_data.append([1,0,0])
            up = up + 1

        elif  float(Decimal(str(bef)) - Decimal(str(aft))) >= float(Decimal("0.001") * Decimal(str(spread))):
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
    spread_tmp_np = np.array(spread_tmp)
    spread_np = np.array(spread_data)
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



    spread_len = len(spread_data)
    print("spread total: ", spread_len)
    if spread_len != 0:
        for k, v in sorted(spread_cnt.items()):
            print(k, v/spread_len)

    return retX, retY, price_np, time_np, close_tmp_np, time_tmp_np, predict_time_np, predict_score_np, end_price_np, spread_np, spread_tmp_np

def get_model():

    if os.path.isfile(model_file):
        model = load_model(model_file)
        #model_gpu = multi_gpu_model(model, gpus=gpu_count)
        model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
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
    spread_trade = {}
    spread_win = {}
    dataX, dataY, price_data, time_data, close, time, predict_time, predict_score, end_price, spread_data, spread_tmp = get_redis_data()
    res = do_predict(dataX,dataY)

    ind5 = np.where(res >=border)[0]
    x5 = res[ind5,:]
    y5= dataY[ind5,:]
    p5 = price_data[ind5]
    t5 = time_data[ind5]
    pt5= predict_time[ind5]
    ps5 = predict_score[ind5]
    ep5 = end_price[ind5]
    sp5 = spread_data[ind5]

    print(t5[0:10])

    Acc = np.mean(np.equal(res.argmax(axis=1),dataY.argmax(axis=1)))
    print("Accuracy over ALL:", Acc)
    print("Total:", len(dataY))
    print("Correct:", len(dataY) * Acc)


    Acc5 = np.mean(np.equal(x5.argmax(axis=1),y5.argmax(axis=1)))
    total_length = len(x5)

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
    #Acc5_up = np.mean(np.equal(x5_up.argmax(axis=1), y5_up.argmax(axis=1)))
    up_cor_length = int(len(np.where(up_eq == True)[0]))
    up_wrong_length = int(up_total_length - up_cor_length)
    print("up_cor_length:"+ str(up_cor_length))
    print("up_wrong_length:" + str(up_wrong_length))

    x5_down = res[down_ind5,:]
    y5_down= dataY[down_ind5,:]
    p5_down = price_data[down_ind5]
    t5_down = time_data[down_ind5]
    down_total_length = len(x5_down)
    #Acc5_down = np.mean(np.equal(x5_down.argmax(axis=1), y5_down.argmax(axis=1)))
    down_eq = np.equal(x5_down.argmax(axis=1), y5_down.argmax(axis=1))
    down_cor_length = int(len(np.where(down_eq == True)[0]))
    down_wrong_length = int(down_total_length - down_cor_length)
    print("down_cor_length:"+ str(down_cor_length))
    print("down_wrong_length:" + str(down_wrong_length))

    cor_list_up_x, cor_list_up_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(up_cor_length) ]), np.ones(up_cor_length, dtype=np.float64)
    wrong_list_up_x, wrong_list_up_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(up_wrong_length) ]), np.ones(up_wrong_length, dtype=np.float64)
    cor_list_down_x, cor_list_down_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(down_cor_length) ]), np.ones(down_cor_length, dtype=np.float64)
    wrong_list_down_x, wrong_list_down_y = np.array([ "yyyy-mm-dd 00:00:00" for i in range(down_wrong_length) ]), np.ones(down_wrong_length, dtype=np.float64)

    money_x, money_y = np.array([ "00:00:00" for i in range(len(time)) ]), np.ones(len(time), dtype=np.float64)
    money_trade_x, money_trade_y = np.array(["00:00:00" for i in range(len(time))]), np.ones(len(time), dtype=np.float64)
    money_not_notice_x, money_not_notice_y = np.array(["00:00:00" for i in range(len(time))]), np.ones(len(time),
                                                                                             dtype=np.float64)
    money_notice_try_x, money_notice_try_y = np.array(["00:00:00" for i in range(len(time))]), np.ones(len(time),
                                                                                             dtype=np.float64)

    money_tmp = {}
    money_trade_tmp = {}
    money_not_notice_tmp = {}
    money_notice_try_tmp = {}

    money = default_money
    money_trade = default_money
    money_not_notice = default_money
    money_notice_try = default_money

    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " Now Calculating")

    cnt_up_cor = 0
    cnt_up_wrong = 0
    cnt_down_cor = 0
    cnt_down_wrong = 0
    loop_cnt = 0
    result_txt = ["time,oandaStartVal,oandaEndVal,predict,probe,predictResult,startVal,endVal,result,correct",]
    trade_cnt = 0
    true_cnt = 0
    trade_win_cnt = 0
    not_notice_win_cnt = 0
    notice_cnt = 0
    trade_wrong_win_cnt = 0
    trade_wrong_lose_cnt = 0
    notice_try_cnt = 0
    notice_try_win_cnt = 0
    not_trade_cnt = 0

    predicts = {}

    r = redis.Redis(host=host, port=6379, db=db_no)
    for x,y,p,t,pt,ps,ep,sp in zip(x5,y5,p5,t5, pt5, ps5, ep5, sp5):

        max = x.argmax()
        probe = str(x[max])

        tradeReult = r.zrangebyscore(symbol + "_TRADE", ps, ps)
        startVal = "NULL"
        endVal = "NULL"
        result = "NULL"
        correct = "NULL"
        #tradeCnt = r.zrangebyscore(symbol + "_TRADE", start_stp, end_stp)
        #print("trade length:", len(tradeCnt))
        if len(tradeReult) != 0:
            trade_cnt = trade_cnt +1
            tmps = json.loads(tradeReult[0].decode('utf-8'))
            startVal = tmps.get("startVal")
            endVal= tmps.get("endVal")
            result = tmps.get("result")
            if result == "win":
                trade_win_cnt = trade_win_cnt +1
                money_trade = money_trade + payout
                money_notice_try = money_notice_try + payout

            else:
                money_trade = money_trade - payoff
                money_notice_try = money_notice_try - payoff
        else:
            #1あとにトレードしている
            tradeReultNotice = r.zrangebyscore(symbol + "_TRADE", ps +1, ps +1)
            """
            if len(tradeReultNotice) == 0:
                tradeReultNotice = r.zrangebyscore(symbol + "_TRADE", ps +2, ps +2)

            if len(tradeReultNotice) == 0:
                tradeReultNotice = r.zrangebyscore(symbol + "_TRADE", ps + 3, ps + 3)

            if len(tradeReultNotice) == 0:
                tradeReultNotice = r.zrangebyscore(symbol + "_TRADE", ps + 4, ps + 4)
            """
            if len(tradeReultNotice) != 0:
                notice_try_cnt = notice_try_cnt + 1
                tmps = json.loads(tradeReultNotice[0].decode('utf-8'))
                #startVal = tmps.get("startVal")
                #endVal = tmps.get("endVal")
                resultNotice = tmps.get("result")
                if resultNotice == "win":
                    notice_try_win_cnt = notice_try_win_cnt + 1
                    money_notice_try = money_notice_try + payout
                else:
                    money_notice_try = money_notice_try - payoff
            else:
                if max == 0 or max == 2:
                    not_trade_cnt += 1
            notice_cnt += 1

        if max == 0 or max == 2:
            res = "win" if max == y.argmax() else "lose"

            flg = False
            for k, v in spread_list.items():
                if sp > v[0] and sp <= v[1]:
                    spread_trade[k] = spread_trade.get(k, 0) + 1
                    if res == "win":
                        spread_win[k] = spread_win.get(k, 0) + 1
                    flg = True
                    break
            if flg == False:
                if sp < 0:
                    spread_trade["spread0"] = spread_trade.get("spread0", 0) + 1
                    if result == "win":
                        spread_win["spread0"] = spread_win.get("spread0", 0) + 1
                else:
                    spread_trade["spread16Over"] = spread_trade.get("spread16Over", 0) + 1
                    if result == "win":
                        spread_win["spread16Over"] = spread_win.get("spread16Over", 0) + 1
        if max == 0:
            # Up predict
            if fx:
                buy = p * fx_position
                sell = ep * fx_position
                # DB上の値は実際の1／100なので100倍している
                profit = float(Decimal(str(sell)) - Decimal(str(buy)) - Decimal(str((0.00001 * fx_spread * fx_position)))) * 100
                money = money + profit
            if max == y.argmax():
                if fx == False:
                    money = money + payout

                    if len(tradeReult) != 0:
                        money_not_notice = money_not_notice + payout
                        not_notice_win_cnt = not_notice_win_cnt + 1
                cor_list_up_x[cnt_up_cor] = pt
                cor_list_up_y[cnt_up_cor] = p
                cnt_up_cor = cnt_up_cor + 1

                if result != "NULL":
                    if result == "win":
                        correct = "TRUE"
                        true_cnt = true_cnt + 1
                    else:
                        correct = "FALSE"
                        trade_wrong_lose_cnt += 1
                result_txt.append(pt + "," + str(p) + "," + str(ep) + "," + "UP" + "," + probe + "," + "win" + "," + startVal + "," + endVal
                                  + "," + result+ "," + correct
                                  )
            else :
                if fx == False:
                    money = money - payoff
                    if len(tradeReult) != 0:
                        money_not_notice = money_not_notice - payoff
                wrong_list_up_x[cnt_up_wrong] = pt
                wrong_list_up_y[cnt_up_wrong] = p
                cnt_up_wrong = cnt_up_wrong + 1

                if result != "NULL":
                    if result == "win":
                        correct = "FALSE"
                        trade_wrong_win_cnt += 1
                    else:
                        correct = "TRUE"
                        true_cnt = true_cnt + 1

                result_txt.append(pt + "," + str(p) + "," + str(ep) + "," + "UP" + "," + probe + "," + "lose" + "," + startVal + "," + endVal
                                  + "," + result+ "," + correct
                                  )
        elif max == 2:
            if fx:
                sell = p * fx_position
                buy =  ep * fx_position
                profit = float(Decimal(str(sell)) - Decimal(str(buy)) - Decimal(str((0.00001 * fx_spread * fx_position)))) * 100
                money = money + profit
            if max == y.argmax():
                if fx == False:
                    money = money + payout
                    if len(tradeReult) != 0:
                        money_not_notice = money_not_notice + payout
                        not_notice_win_cnt = not_notice_win_cnt + 1

                cor_list_down_x[cnt_down_cor] = pt
                cor_list_down_y[cnt_down_cor] = p
                cnt_down_cor = cnt_down_cor + 1

                if result != "NULL":
                    if result == "win":
                        correct = "TRUE"
                        true_cnt = true_cnt + 1
                    else:
                        correct = "FALSE"
                        trade_wrong_lose_cnt += 1

                result_txt.append(pt + "," + str(p) + "," + str(ep) + "," + "DOWN" + "," + probe + "," + "win" + "," + startVal + "," + endVal
                                  + "," + result+ "," + correct
                                  )

            else:
                if fx == False:
                    money = money - payoff
                    if len(tradeReult) != 0:
                        money_not_notice = money_not_notice - payoff
                wrong_list_down_x[cnt_down_wrong] = pt
                wrong_list_down_y[cnt_down_wrong] = p
                cnt_down_wrong = cnt_down_wrong + 1

                if result != "NULL":
                    if result == "win":
                        correct = "FALSE"
                        trade_wrong_win_cnt += 1
                    else:
                        correct = "TRUE"
                        true_cnt = true_cnt + 1

                result_txt.append(pt + "," + str(p) + "," + str(ep) + "," + "DOWN" + "," + probe + "," + "lose" + "," + startVal + "," + endVal
                                  + "," + result+ "," + correct
                                  )

        money_tmp[pt] = money
        money_trade_tmp[pt] = money_trade
        money_not_notice_tmp[pt] = money_not_notice
        money_notice_try_tmp[pt] = money_notice_try
        loop_cnt = loop_cnt + 1

    prev_money = default_money
    prev_trade_money = default_money
    prev_not_notice_money = default_money
    prev_notice_try_money = default_money
    #T = time[0]
    #print("T:" + T[11:])
    for i, ti in enumerate(time):
        if ti in money_tmp.keys():
            prev_money = money_tmp[ti]

        money_x[i] = ti[11:13]
        money_y[i] = prev_money

    for i, ti in enumerate(time):
        if ti in money_trade_tmp.keys():
            prev_trade_money = money_trade_tmp[ti]

        money_trade_x[i] = ti[11:13]
        money_trade_y[i] = prev_trade_money

    for i, ti in enumerate(time):
        if ti in money_not_notice_tmp.keys():
            prev_not_notice_money = money_not_notice_tmp[ti]

        money_not_notice_x[i] = ti[11:13]
        money_not_notice_y[i] = prev_not_notice_money

    for i, ti in enumerate(time):
        if ti in money_notice_try_tmp.keys():
            prev_notice_try_money = money_notice_try_tmp[ti]

        money_notice_try_x[i] = ti[11:13]
        money_notice_try_y[i] = prev_notice_try_money

    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " Now Plotting")
    fig = plt.figure()
    #価格の遷移
    ax1 = fig.add_subplot(2,1,1)
    #ax1.plot(time,close)
    ax1.plot(close, 'g')

    """
    ax1.plot(cor_list_up_x, cor_list_up_y, 'b^')
    ax1.plot(wrong_list_up_x, wrong_list_up_y, 'r^')
    ax1.plot(cor_list_down_x, cor_list_down_y, 'bv')
    ax1.plot(wrong_list_down_x, wrong_list_down_y, 'rv')
    """

    ax2 = ax1.twinx()
    ax2.plot(money_y)
    ax2.plot(money_trade_y,"r")
    ax2.plot(money_not_notice_y, "y")
    ax2.plot(money_notice_try_y, "m")

    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(spread_tmp, 'g')

    index = np.arange(0,len(money_x),3600// int(s))
    plt.xticks(index,money_x[index])

    for txt in result_txt:
        res = txt.find("FALSE")
        if res != -1:
            print(txt)

    #print('\n'.join(result_txt))

    print("trade cnt: " + str(trade_cnt))
    print("trade correct: " + str(true_cnt / trade_cnt))
    print("trade wrong cnt: " + str(trade_cnt - true_cnt))
    print("trade wrong win cnt: " + str(trade_wrong_win_cnt))
    print("trade wrong lose cnt: " + str(trade_wrong_lose_cnt))

    print("not_trade cnt: " + str(not_trade_cnt))
    print("notice cnt: " + str(notice_cnt))
    print("not_notice accuracy: " + str(not_notice_win_cnt / trade_cnt))
    print("not_notice money: " + str(prev_not_notice_money))

    print("notice_try accuracy: " + str((trade_win_cnt + notice_try_win_cnt) / (trade_cnt + notice_try_cnt)))
    print("notice_try money: " + str(prev_notice_try_money))
    print("notice_try_cnt: " + str(notice_try_cnt))

    print("trade accuracy: " + str(trade_win_cnt/trade_cnt))
    print("trade money: " + str(prev_trade_money))

    print("predict money: " + str(money))

    tmp_acc = (up_cor_length + down_cor_length)/(len(up_ind5) + len(down_ind5))
    print("Accuracy over " + str(border) + ":", tmp_acc)
    print("Total:", len(up_ind5) + len(down_ind5))
    print("Correct:", (len(up_ind5) + len(down_ind5)) * tmp_acc)
    print("trade cnt rate: " + str(trade_cnt/(len(up_ind5) + len(down_ind5))))

    for k, v in sorted(spread_list.items()):
        if spread_trade.get(k, 0) != 0:
            print(k, " cnt:", spread_trade.get(k, 0), " win rate:", spread_win.get(k, 0) / spread_trade.get(k))
        else:
            print(k, " cnt:", spread_trade.get(k, 0))

    #plt.title('border:' + str(border) + " payout:" + str(payout) + " except index:" + str(except_index))
    plt.show()