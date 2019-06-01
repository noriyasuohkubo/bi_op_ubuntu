import os
from datetime import datetime
from datetime import timedelta
from keras.utils.training_utils import multi_gpu_model
import time

#定数ファイル
host = "127.0.0.1"

symbol = "GBPJPY"
symbols = [symbol]
#symbols = [symbol + "5", symbol + "10",symbol]

export_host = "ubuntu4"
suffix = ".70*8"

border = 0.56

border_spread = 0.008
limit_border_flg = False

spread = 1

except_list = [20,21,22]

#for usdjpy
#except_list = [1,8,10,11,12,17,18,19,20,21,22]

#for gbpjpy aus
#except_list = [4,7,8,9,10,11,12,13,14,15,16,17,18,19,20, 21, 22]
#except_list = [4,7,8,9,10,11,12,13,14,15,16,17,20, 21, 22]

#for gbpjpy opt
#except_list = [4,6,7,8,9,10,11,12,13,14,15,16,17,20, 21, 22]

#for gbpjpy snc
#except_list = [3,4,6,7,8,9,10,11,13,14,15,16,17,20, 21, 22]

payout = 950*10
payoff = 1000*10

start = datetime(2019, 4, 30,22)
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2019, 6, 1 )
end_stp = int(time.mktime(end.timetuple()))

maxlen = 400
pred_term = 15
s = "2"

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

spread_list = {"spread0":(-1,0.000),"spread2":(0.000,0.002), "spread4":(0.002,0.004),"spread6":(0.004,0.006),"spread8":(0.006,0.008)
    , "spread10": (0.008, 0.010), "spread12": (0.010, 0.012), "spread14": (0.012, 0.014), "spread16": (0.014, 0.016),"spread16Over":(0.016,1),}

db_nos = {"ubuntu1":11,"ubuntu2":12,"ubuntu3":13,"ubuntu4":14,"ubuntu4-2":10,"ubuntu5":15}

db_no = db_nos[export_host]

db_suffix_trade_list = {"ubuntu1":"","ubuntu2":"","ubuntu3":"_OPT","ubuntu4":"","ubuntu4-2":"","ubuntu5":""}

#db_suffixs = (1,2,3,4,5)
#db_suffixs = ("",)
db_suffix = db_suffix_trade_list[export_host]

model_dir = "/app/bin_op/model"
gpu_count = 1
batch_size = 8192* gpu_count

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

