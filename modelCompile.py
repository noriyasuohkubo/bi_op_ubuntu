from flask import Flask, request
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

#GPU使わない方がはやい
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
app = Flask(__name__)

#symbol = "EURUSD"
symbol = "GBPJPY"

db_no = 7

maxlen = 300
drop = 0.1
in_num=1
pred_term = 15
s = "2"

suffix = ".41*15"

bin_type = ""
spread = 1
if bin_type == "_spread":
    bin_type = bin_type + str(spread -1)

np.random.seed(0)
n_hidden =  30
n_hidden2 = 0
n_hidden3 = 0
n_hidden4 = 0

askbid = "_bid"

current_dir = os.path.dirname(__file__)
ini_file = os.path.join(current_dir,"config","config.ini")
config = configparser.ConfigParser()
config.read(ini_file)
MODEL_DIR = config['lstm']['MODEL_DIR']

logging.config.fileConfig( os.path.join(current_dir,"config","logging.conf"))
logger = logging.getLogger("app")

file_prefix = symbol + "_bydrop_in" + str(in_num) + "_" + s + "_m" + str(maxlen) + "_term_" + str(pred_term * int(s)) + "_hid1_" + str(n_hidden) + \
                          "_hid2_" + str(n_hidden2) + "_hid3_" + str(n_hidden3) + "_hid4_" + str(n_hidden4) + "_drop_" + str(drop) + askbid + bin_type
print("MODEL:", file_prefix +".hdf5" + suffix)
model_file = os.path.join(MODEL_DIR, file_prefix +".hdf5" + suffix)



# model and backend graph must be created on global
global model, graph
if os.path.isfile(model_file):
    model = load_model(model_file)
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    model.save(model_file)

    #json_string = model.to_json()
    #text_file = open("/tmp/model.json", "w")
    #text_file.write(json_string)
    #text_file.close()
    #model.save_weights('/app/bin_op/model/model_weights.h5')
    print('Model saved')
else:
    logger.warning("no model exists")


