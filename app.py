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
from flask import Flask, request
import subprocess
import send_mail as m
from datetime import datetime
from datetime import date

#nginxとflaskを使ってhttpによりAiの予想を呼び出す方式
#systemctl start nginxでwebサーバを起動後、以下のコマンドによりuwsgiを起動し、localhost:80へアクセス
#uwsgi --ini /app/bin_op/uwsgi.ini


machine = "amd6"
maxlen = 600;
model_dir = "/app/bin_op/model"
file_prefix = "GBPJPY_lstm_close_divide_2_m600_term_30_hid1_60_hid2_0_hid3_0_hid4_0_drop_0.0_bid_merg_2_set_ALL.hdf5.90*17"
model_file = os.path.join(model_dir, file_prefix)

app = Flask(__name__)

if os.path.isfile(model_file):
    model = load_model(model_file)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("load model")
else:
    msg = "the Model not exists!"
    print(msg)
    m.send_message("uwsgi " + machine + " ", msg)


def do_predict(retX):
    #print(retX.shape)
    #start = time.time()
    res = model.predict(retX, verbose=0,batch_size=1 )

    #elapsed_time = time.time() - start
    #print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    #print(res)

    #K.clear_session()
    return res


@app.route("/", methods=['GET', 'POST'])
def hello():
    data = request.get_json()
    closes = data["closes"]

    #print(closes[:11])

    close = 10000 * np.log(closes / shift(closes, 1, cval=np.NaN))[1:]
    retX = np.zeros((1, maxlen, 1))
    retX[:, :, 0] = close[:]
    res = do_predict(retX)
    res_str = ""

    res_str = str(res[0][0]) + "," + str(res[0][1]) + "," + str(res[0][2])
    #print(res_str)
    return res_str

if __name__ == "__main__":
    app.run()

