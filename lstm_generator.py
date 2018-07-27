import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import initializers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import redis
import traceback
import json
import time
import pandas as pd
from sklearn import preprocessing
from keras.models import load_model
from keras import backend as K
import os
from keras.callbacks import CSVLogger
import configparser
from keras.callbacks import ModelCheckpoint
from scipy.ndimage.interpolation import shift
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop, Nadam
from logging import getLogger
from datetime import datetime
from datetime import timedelta
import time
from decimal import Decimal
from DataSequence import DataSequence
import multiprocessing
from keras.backend import tensorflow_backend


config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)

logger = getLogger(__name__)

gpu_count = 2
type = "category"

symbol = "GBPJPY"

train = True

maxlen = 300
pred_term = 10

epochs = 2
#batch_size = 8192 * gpu_count
batch_size = 100

askbid = "_bid"
s = "3"
drop = 0.1

in_num = 1
np.random.seed(0)

spread = 1
n_hidden = 30
n_hidden2 = 0
n_hidden3 = 0
n_hidden4 = 0

file_prefix = symbol + "_bydrop_in" + str(in_num) + "_" + s + "_m" + str(maxlen) + "_term_" + str(
    pred_term * int(s)) + "_hid1_" + str(n_hidden) + \
              "_hid2_" + str(n_hidden2) + "_hid3_" + str(n_hidden3) + "_hid4_" + str(n_hidden4) + "_drop_" + str(drop) + askbid

current_dir = os.path.dirname(__file__)
ini_file = os.path.join(current_dir, "config", "config.ini")
history_file = os.path.join(current_dir, "history", file_prefix + "_history.csv")

config = configparser.ConfigParser()
config.read(ini_file)

SYMBOL_DB = json.loads(config['lstm']['SYMBOL_DB'])
MODEL_DIR = config['lstm']['MODEL_DIR']

model_file = os.path.join(MODEL_DIR, file_prefix + ".hdf5")

#process_count = multiprocessing.cpu_count() - 1
process_count = 1
print("process_count:" , process_count)

def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)

def mean_pred(y_true, y_pred):
    tmp = y_true * y_pred
    o = tf.constant(1, dtype=tf.float32)
    z = tf.constant(0, dtype=tf.float32)

    return K.mean(tf.map_fn(lambda x: tf.cond(tf.greater_equal(x[0], z), lambda: o, lambda: z), tmp))


def create_model(n_in=in_num, n_out=3):
    model = None

    with tf.device("/cpu:0"):
        if type == "category":
            model = Sequential()
            """
            model.add(LSTM(n_hidden,input_shape=(maxlen, n_in)
                           , kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)))
                           #,return_sequences = True))
            """
            model.add(Bidirectional(LSTM(n_hidden
                                         , kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)
                                         )
                                    # ,return_sequences=True)
                                    , input_shape=(maxlen, n_in)
                                    ))
            model.add(Dropout(drop))
            """
            model.add(Bidirectional(LSTM(n_hidden2
                                         ,kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)
                                         )
                                         #, return_sequences=True)
                           ))
            model.add(Dropout(drop))

            model.add(Bidirectional(LSTM(n_hidden3
                                         ,kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None))
                           ))
            model.add(Dropout(drop))


            """
            model.add(Dense(n_out, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=1, seed=None)))
            model.add(Activation('softmax'))

        elif type == 'mean':
            model = Sequential()
            model.add(LSTM(n_hidden,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                           input_shape=(maxlen, n_in)
                           , return_sequences=True))
            model.add(LSTM(n_hidden,
                           kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
                           ))
            model.add(Dense(n_out, kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
            model.add(Activation('linear'))

    return model


def get_model():
    model = None
    if os.path.isfile(model_file):
        model = load_model(model_file, custom_objects={"mean_pred": mean_pred})
        print("Load Model")
    else:
        model = create_model()
    model_gpu = model
    #model_gpu = multi_gpu_model(model, gpus=gpu_count)
    if type == "category":
        model_gpu.compile(loss='categorical_crossentropy',
                          optimizer='rmsprop', metrics=['accuracy'])
    elif type == 'mean':
        model_gpu.compile(loss='mean_squared_error',
                          optimizer="rmsprop", metrics=[mean_pred])
    return model, model_gpu


'''

モデル学習
'''


# callbacks.append(CSVLogger("history.csv"))
# look
# https://qiita.com/yukiB/items/f45f0f71bc9739830002

def do_train():
    model, model_gpu = get_model()
    early_stopping = EarlyStopping(monitor='loss', patience=100, verbose=1)
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(
            MODEL_DIR,
            'lstm_{epoch:03d}_s' + s + '.hdf5'),
        save_best_only=False)
    dataSequence = DataSequence(maxlen, pred_term, s, in_num, batch_size, symbol)

    hist = model_gpu.fit_generator(dataSequence,
                                    steps_per_epoch=dataSequence.__len__(),
                                    epochs=epochs,
                                    max_queue_size=process_count * 1,
                                    callbacks=[early_stopping, CSVLogger(history_file)],
                                    use_multiprocessing=True, workers=process_count
                                   )


    # save model, not model_gpu
    # see http://tech.wonderpla.net/entry/2018/01/09/110000
    model.save(model_file)
    print('Model saved')

    if hist is not None:
        # 損失の履歴をプロット
        plt.plot(hist.history['loss'])
        plt.title('model loss')
        plt.show()

    K.clear_session()

    print("END")



if __name__ == '__main__':
    do_train()
