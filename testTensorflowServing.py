import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import numpy as np
import time

#test for tensorflow serving
#see: https://qiita.com/FukuharaYohei/items/13ea650c6f370111b0ba

#make test data
maxlen = 600
in_num = 1

tmpF = float(1)
floatArr = []
for i in range(600):
    floatArr.append(float(tmpF))
    tmpF = float(tmpF) + float(1)

print(floatArr)


#gRPC test
#cmd: tensorflow_model_server --port=9000 --model_name=saved_model --model_base_path=/app/bin_op/saved_model
request = predict_pb2.PredictRequest()
request.model_spec.name = 'saved_model'  #モデル名
request.model_spec.signature_name = 'serving_default'  #signature

request.inputs['lstm_1_input'].CopyFrom(
    #inputは1次元しかできない
    tf.compat.v1.make_tensor_proto(floatArr, shape=[1,maxlen,in_num]))
start = time.time()
channel = grpc.insecure_channel("localhost:9000")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
print(stub.Predict(request, 10))   #10秒のtimeout
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")


request = predict_pb2.PredictRequest()
request.model_spec.name = 'saved_model'  #モデル名
request.model_spec.signature_name = 'serving_default'  #signature

request.inputs['lstm_1_input'].CopyFrom(
    #inputは1次元しかできない
    tf.compat.v1.make_tensor_proto(floatArr, shape=[1,maxlen,in_num]))
start = time.time()
channel = grpc.insecure_channel("localhost:9000")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
print(stub.Predict(request, 10))   #10秒のtimeout
elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

#elapsed_time:0.044168710708618164[sec]