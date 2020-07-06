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
#CPUのスレッド数を制限してロードアベレージの上昇によるハングアップを防ぐ
os.environ["OMP_NUM_THREADS"] = "3"

fx = False
fx_position = 10000
fx_spread = 1

current_dir = os.path.dirname(__file__)

logging.config.fileConfig( os.path.join(current_dir,"config","logging.conf"))
logger = logging.getLogger("app")

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

def do_predict(test_X):

    model = get_model()
    if model is None:
        return None
    start = time.time()
    res = model.predict(test_X, verbose=0, )
    end = time.time()
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    start = time.time()
    res = model.predict(test_X, verbose=0, )
    end = time.time()
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    start = time.time()
    res = model.predict(test_X, verbose=0,batch_size=1 )
    end = time.time()
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    print("Predict finished")

    K.clear_session()
    return res


if __name__ == "__main__":
    spread_trade = {}
    spread_win = {}
    tmpF = 1
    floatArr = []
    for i in range(600):
        floatArr.append(tmpF)
        tmpF = tmpF + 1

    print(floatArr)
    retX = np.zeros((1, maxlen, in_num))
    retX[:, :, 0] = floatArr[:]

    res = do_predict(retX)
    print(str(res[0][0]))
    print(res)
