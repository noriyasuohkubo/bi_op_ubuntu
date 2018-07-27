from keras.utils import Sequence
from pathlib import Path
import pandas
import numpy as np
from keras.utils import np_utils
from datetime import datetime
import time
from decimal import Decimal
from scipy.ndimage.interpolation import shift
import redis
import json
import random

db_no = 3

start = datetime(2018, 2, 1)
start_score = int(time.mktime(start.timetuple()))

end = datetime(2000, 1, 1)
end_score = int(time.mktime(end.timetuple()))

except_highlow = True

except_list = [20, 21, 22]

spread = 1

class DataSequence(Sequence):
    def __init__(self, maxlen, pred_term, s, in_num, batch_size, symbol):
        #コンストラクタ
        self.maxlen = maxlen
        self.pred_term = pred_term
        self.s = s
        self.rec_num = 15000000 + maxlen + pred_term + 1
        self.in_num = in_num
        self.batch_size = batch_size
        self.symbol = symbol

        print("DB_NO:", db_no)
        r = redis.Redis(host='localhost', port=6379, db=db_no)
        result = r.zrevrangebyscore(symbol, start_score, end_score, start=0, num=self.rec_num + 1)

        close_tmp  = []
        time_tmp = []

        result.reverse()
        for line in result:
            tmps = json.loads(line.decode('utf-8'))
            close_tmp.append(tmps.get("close"))
            time_tmp.append(tmps.get("time"))

        self.close = 10000 * np.log(close_tmp/shift(close_tmp, 1, cval=np.NaN) )[1:]
        self.train_list = []

        print(self.close[0:10])
        print(close_tmp[0:10])
        print(time_tmp[-5:])
        print(time_tmp[0:5])

        up =0
        same =0
        tmp_data_length = len(self.close) - maxlen - pred_term -1

        for i in range(tmp_data_length):

            #ハイローオーストラリアの取引時間外を学習対象からはずす
            if except_highlow:

                if datetime.strptime(time_tmp[1 + i + maxlen -1], '%Y-%m-%d %H:%M:%S').hour in except_list:
                    continue;

            bef = close_tmp[1 + i + maxlen -1]
            aft = close_tmp[1 + i + maxlen + pred_term -1]

            tmp_label = None
            if float(Decimal(str(aft)) - Decimal(str(bef))) >= 0.00001 * spread:
                # 上がった場合
                tmp_label = [1, 0, 0]
                up = up + 1
            elif float(Decimal(str(bef)) - Decimal(str(aft))) >= 0.00001 * spread:
                tmp_label = [0, 0, 1]
            else:
                tmp_label = [0, 1, 0]
                same = same + 1

            self.train_list.append([i, tmp_label])

        cut_num = len(self.train_list) % self.batch_size
        print("tmp train list length: " , str(len(self.train_list)))
        print("cut_num: " , str(cut_num))

        self.train_list = self.train_list[cut_num:]
        print("train list length: " , str(len(self.train_list)))

        self.data_length = len(self.train_list)
        self.steps_per_epoch = int(self.data_length / self.batch_size)
        print("steps_per_epoch: ", self.steps_per_epoch)
        print("UP: ",up/self.data_length)
        print("SAME: ", same / self.data_length)
        print("DOWN: ", (self.data_length - up - same) / self.data_length)

        print(self.train_list[0:10])
        print(self.close[self.train_list[0][0]:(self.train_list[0][0] + self.maxlen)])
        print(self.train_list[0][1])

    def __getitem__(self, idx):
        # データの取得実装
        close_data, label_data = [], []
        #print("idx:"+ str(idx))
        start_idx = idx * self.batch_size
        for i in range(self.batch_size):
            target_idx = self.train_list[start_idx + i][0]
            label_data.append(self.train_list[start_idx + i][1])

            close_data.append(self.close[target_idx:(target_idx + self.maxlen)])


        close_np = np.array(close_data)

        retX = np.zeros((len(close_np), self.maxlen, self.in_num))
        retX[:, :, 0] = close_np[:]
        retY = np.array(label_data)

        if idx == 0:
            print("X SHAPE:", retX.shape)
            print("Y SHAPE:", retY.shape)
            print("X :", retX[0:10])
            print("Y :", retY[0:10])

        return retX, retY

    def __len__(self):
        # １エポック中のステップ数
        return self.steps_per_epoch

    def on_epoch_end(self):
        # epoch終了時の処理 リストをランダムに並べ替える
        random.shuffle(self.train_list)
        print(self.train_list[0:10])

    def get_data_length(self):

        return self.data_length