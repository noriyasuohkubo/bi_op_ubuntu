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
suffix = ".41*15"

spread = 1

#for gbpjpy
#except_list = [4,7,8,9,10,11,12,13,14,15,16,17,20, 21, 22]
except_list = [20, 21, 22]

payout = 950
payoff = 1000

start = datetime(2018, 12, 12,22)
start_stp = int(time.mktime(start.timetuple()))

end = datetime(2018, 12, 14 )
end_stp = int(time.mktime(end.timetuple()))

maxlen = 300
pred_term = 15
s = "2"

merg = ""
merg_file = ""
if merg != "":
    merg_file = "_merg_" + str(merg)

n_hidden = 30
n_hidden2 = 0
n_hidden3 = 0
n_hidden4 = 0

drop = 0.1
in_num = 1
spread = 1

db_nos = {"ubuntu1":11,"ubuntu2":12,"ubuntu3":13,"ubuntu4":14,"ubuntu5":15}

db_no = db_nos[export_host]

db_suffix_trade_list = {"ubuntu1":"","ubuntu2":"","ubuntu3":"_OPT","ubuntu4":"","ubuntu5":"_SNC"}

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
