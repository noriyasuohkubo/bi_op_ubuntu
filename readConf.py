import os
from datetime import datetime
from datetime import timedelta
from keras.utils.training_utils import multi_gpu_model
import time
os.environ["OMP_NUM_THREADS"] = "3"
#定数ファイル
host = "127.0.0.1"

symbol = "GBPJPY"
symbols = [symbol]
#symbols = [symbol + "5", symbol + "10",symbol]

export_host = "noriyasu"
#取引制限ありと想定して予想させる
#この場合、必ずしも予想時に実際のトレードがされると限らないので、トレード実績を見たい場合はFalseにする
restrict_flg = True

#THE OPTIONである
the_option_flg = False

#TRB or SPR
type = "SPR"

#デモである
demo = False

suffix = ".70*8"

border = 0.58

#except_list = [20,21,22]
except_list = [21,22,23]

border_spread = 0.008
limit_border_flg = False

spread = 1

#for usdjpy
#except_list = [1,8,10,11,12,17,18,19,20,21,22]

#for gbpjpy aus
#except_list = [4,7,8,9,10,11,12,13,14,15,16,17,18,19,20, 21, 22]
#except_list = [4,7,8,9,10,11,12,13,14,15,16,17,20, 21, 22]

#for gbpjpy opt
#except_list = [4,6,7,8,9,10,11,12,13,14,15,16,17,20, 21, 22]

#for gbpjpy snc
#except_list = [3,4,6,7,8,9,10,11,13,14,15,16,17,20, 21, 22]

start = datetime(2019, 9, 1, 22)
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2020, 1, 1, 22 )
end_stp = int(time.mktime(end.timetuple()))

maxlen = 400
pred_term = 15
s = "2"

#trb 30:1000, 60:950. 180:900, 500:850
#spr 30:1200, 60:1100, 180:1050, 300:1000
payout = 1000
payoff = 1000

merg = ""
merg_file = ""
if merg != "":
    merg_file = "_merg_" + str(merg)

n_hidden = 40
n_hidden2 = 0
n_hidden3 = 0
n_hidden4 = 0

drop = 0.0
in_num = 1
spread = 1

spread_list = {"spread0":(-1,0.000),"spread1":(0.000,0.001),"spread2":(0.001,0.002),"spread3":(0.002,0.003), "spread4":(0.003,0.004)
    ,"spread5":(0.004,0.005),"spread6":(0.005,0.006),"spread7":(0.006,0.007),"spread8":(0.007,0.008)
    , "spread10": (0.008, 0.010), "spread12": (0.010, 0.012), "spread14": (0.012, 0.014), "spread16": (0.014, 0.016),"spread16Over":(0.016,1),}

db_nos = {"opt":1,"snc":1,"noriyasu":11,"yorioko":14,"kazuo":8,"fil":14}

db_no = db_nos[export_host]

#db_suffix_trade_list = {"ubuntu1":"","ubuntu2":"","ubuntu3":"_OPT","ubuntu4":"","ubuntu4-2":"","ubuntu5":"","ubuntu18":""}

#db_suffixs = (1,2,3,4,5)
#db_suffixs = ("",)
#db_suffix = db_suffix_trade_list[export_host]

if demo:
    db_suffix = "_DEMO"
else:
    db_suffix = ""

db_key = symbol + "_" + str(int(s) * pred_term)  + "_" + type
#db_key = symbol

if(the_option_flg):
    db_key = db_key + "_OPT"

db_key_trade = db_key + "_TRADE"
print("db_key: " + db_key)


model_dir = "/app/bin_op/model"
gpu_count = 1
batch_size = 2048* gpu_count

except_index = False
except_highlow = True

#process_count = multiprocessing.cpu_count() - 1
process_count = 1
askbid = "_bid"
type = "category"

default_money = 0

current_dir = os.path.dirname(__file__)

file_prefix = symbol + "_bydrop_in" + str(in_num) + "_" + s + "_m" + str(maxlen) + "_term_" + str(pred_term * int(s)) + "_hid1_" + str(n_hidden) + \
                          "_hid2_" + str(n_hidden2) + "_hid3_" + str(n_hidden3) + "_hid4_" + str(n_hidden4) + "_drop_" + str(drop)  + askbid + merg_file

history_file = os.path.join(current_dir, "history", file_prefix + "_history.csv")
model_file = os.path.join(model_dir, file_prefix + ".hdf5" + suffix)
print("Model is ", model_file)

