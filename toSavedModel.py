import tensorflow as tf

#tensorflow servingで使用するモデルをkerasのモデルからconvertする
#see: https://qiita.com/t_shimmura/items/1ebd2414310f827ed608

version = 1

model = tf.keras.models.load_model('/app/bin_op/model/GBPJPY_lstm_close_divide_2_m600_term_30_hid1_60_hid2_0_hid3_0_hid4_0_drop_0.0_bid_merg_2_set_ALL.hdf5.90*17.compiled')
export_path = '/app/bin_op/saved_model' + str(version)

with tf.keras.backend.get_session() as sess:

    print(model.input)
    for t in model.outputs:
        print(t)
    for i in range(len(model.outputs)):
        print(i)

    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'lstm_1_input': model.input},
        outputs={"output_node" + str(i):t for i,t in enumerate(model.outputs)})


#after make savedModel ,check the result with below command!!
#saved_model_cli show --all --dir /app/bin_op/saved_model/1

#start tensorflow serving
#cmd: tensorflow_model_server --port=9000 --model_name=saved_model --model_base_path=/app/bin_op/saved_model
