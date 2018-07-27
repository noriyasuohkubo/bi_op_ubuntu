import tensorflow as tf
import os

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph
tfmodel = load_graph("/app/bin_op/tensorflow_model/GBPJPY_bydrop_in1_5_m100_term_30_hid1_30_hid2_0_hid3_0_hid4_0_drop_0.pb")

# print operations in the model, in tf.Operation format

opers = tfmodel.get_operations()
print(opers)

inLayer = tfmodel.get_operation_by_name('import/bidirectional_1_input')
outLayer = tfmodel.get_operation_by_name('import/output_node0')


print(inLayer.outputs[0])
print(outLayer.outputs[0])