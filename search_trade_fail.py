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
from bi_op_ubuntu.indices import index
from decimal import Decimal

#指定開始日時から1分ずつ未来へずれて,指定DBのレコードが存在するか調べ
#存在しない日(トレード時間中にエラー停止した日)を表示する

#DB名
symbol = "GBPJPY_30_SPR"

db_no = 8

host = "127.0.0.1"

start = datetime(2020, 4, 15, 23, 5)
end = datetime(2020, 8, 21, 19, 45)

"""
tmp_start = start + timedelta(hours=1)
print(tmp_start.hour)
print(start.hour)
"""

redis_db = redis.Redis(host=host, port=6379, db=db_no, decode_responses=True)

#失敗した日の年、月、日をリストに追加する
# example:[[2020,1,2],[2020,2,3]]
#月曜日が0で日曜日が6なので、4,5で失敗しても追加しない
fail_list = []



while True:
    if start.weekday() == 4 or start.weekday() == 5:
        start = start + timedelta(days=1)
        continue
    prev_start = start
    tmp_list = [prev_start.year, prev_start.month, prev_start.day]
    tmp_end = prev_start + timedelta(hours=20) +timedelta(minutes=40)
    #print(start.month,start.day,start.hour,start.minute)
    #print(tmp_end.month,tmp_end.day,tmp_end.hour,tmp_end.minute)
    while True:
        stp = int(time.mktime(start.timetuple()))
        result_data = redis_db.zrangebyscore(symbol, stp, stp, withscores=False)
        if len(result_data) == 0:
            fail_list.append(tmp_list)
            break
        start = start + timedelta(minutes=1)
        if tmp_end < start:
            break

    start = prev_start + timedelta(days=1)

    if end < prev_start:
        break

for i in fail_list:
    print(str(i[0]) + "/" + str(i[1]).zfill(2) + "/" + str(i[2]).zfill(2) + ",")