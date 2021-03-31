from keras.models import Model

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Input, Reshape, Conv1D
from keras.optimizers import Adam
from keras import callbacks
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.python.client import timeline
from data_loader_2 import dataset
import numpy as np
import random
import time


# def vec_R_array(vec):

data = dataset('./cov1')
data_list_all_ = data.train_data_pair  #512频点的混合数据，每个为一个pair，格式为（频点序号，矩阵的64维向量压缩）
data_list_ = [i for i in data_list_all_ if(i[0]<192 or i[0]>=320)] #train data
print(len(data_list_all_))
db_gain = 0
gain_cnt = 0
total_cnt = 0
inputs=Input(shape=[1,])

model = Dense(32, activation='relu')(inputs)

#model = Conv2D(8, (3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(8, 8, 4))(model)
model = BatchNormalization(axis=-1)(model)
model = Reshape([32,1])(model)    #第二次reshape
model = Conv1D(1,kernel_size=8, strides=4, activation='relu',padding='same')(model)
model = Flatten()(model)
model = Dense(128,activation='relu')(model)
model = BatchNormalization(axis=-1)(model)
outputs = Dense(64)(model)
model = Model(inputs=inputs, outputs=outputs)

adam =  Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
callbacks_list = [callbacks.TensorBoard(log_dir='./logs', write_graph=True, write_images=False)]
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
model.compile(loss='mse',optimizer=adam,options=run_options,run_metadata=run_metadata)

tick = time.time()
for i in range(0,10):
    #for a in [0,45]:


    a = random.randint(0,int(len(data_list_all_)/512-1))
    data_list_all = data_list_all_[a*512:a*512+512]

    data_list_all_data =np.array([i[0] for i in data_list_all]).astype(np.float64)
    data_list_all_label = np.array([i[1] for i in data_list_all]).astype(np.float64)


    data_list = data_list_[a*384:a*384+384]
    random.shuffle(data_list)

    train = data_list[:320]
    test = data_list[-64:]
    train_data = np.array([i[0] for i in train])
    train_data = train_data.astype(np.float64)#.reshape(1,1,-1)
    train_data = train_data/512
    #print(train_data.shape)


    train_label = np.array([i[1] for i in train])
    train_label =train_label.astype(np.float64)
    norm_max = train_label.max()
    #print(norm_max)
    train_label = train_label/norm_max

    test_data = np.array([i[0] for i in test])
    test_data = test_data.astype(np.float64)#.reshape(1,1,-1)
    test_data = test_data/512

    test_label = np.array([i[1] for i in test])
    test_label = test_label.astype(np.float64)
    test_label = test_label/norm_max


    array_in_true = np.mat(data.R_in_oracle[a*64:a*64+64]).reshape(8,8)
    array_sin = np.mat(data.vec_to_array(data_list_all_label[256])).reshape(8,8)
    SINR_old = np.log10((np.diag(array_sin) - np.diag(array_in_true)) / np.diag(array_in_true)) * 10.0

    RR_r = array_in_true.I * (array_sin - array_in_true)
    ei_r, eiv_r = np.linalg.eig(RR_r)
    marg = np.argmax(ei_r)
    w_r = np.mat(np.reshape(eiv_r[:, marg], (8, 1)))
    p_IN_r = np.real((w_r.H * array_in_true * w_r)[0, 0])
    p_SIN_r = np.real((w_r.H * array_sin * w_r)[0, 0])
    SINR_r = np.log10((p_SIN_r - p_IN_r) / p_IN_r) * 10.0
    if(SINR_r<12 or np.mean(np.real(SINR_old))>0):
        continue
    #model.load_weights('MLP.weights.best.hdf5')
    #checkpoint = ModelCheckpoint(filepath='MLP.weights.best.hdf5', verbose=0, save_best_only=True)
    #model.fit(train_data, train_label, validation_data=(test_data, test_label), epochs=50, batch_size=32,
    #          callbacks=[checkpoint],shuffle=True, verbose=0)
    model.fit(train_data, train_label, validation_data=(test_data, test_label), epochs=50, batch_size=32,
           shuffle=True, verbose=0,callbacks=callbacks_list)

    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open(str(a)+'_timeline.json', 'w') as f:
        f.write(ctf)
    print('timeline.json has been saved!')
    #model.load_weights('MLP.weights.best.hdf5')

    #data_list_all_label  长度

    pred_output = model.predict(data_list_all_data[256].reshape(1,))
    array_in_pred = np.mat(data.vec_to_array(pred_output[0])).reshape(8,8)*norm_max

    # RR_r = R_in_r.I

    # print(array_sin*80000)
    # print('-----------------')
    # print(array_in_true*80000)
    # print('------------------')
    # print(array_in_pred*80000)

    RR_p = array_in_pred.I * (array_sin - array_in_pred)
    # RR_p = R_in_p.I
    ei_p, eiv_p = np.linalg.eig(RR_p)
    marg = np.argmax(ei_p)
    w_p = np.mat(np.reshape(eiv_p[:, marg], (8, 1)))



    pwr_p = (w_p.H * array_in_true * w_p)[0, 0].real
    pwr_r = (w_r.H * array_in_true * w_r)[0, 0].real
    pwr_ant = np.mean(np.real(np.diag(array_in_true)))


    p_IN_p = np.real((w_p.H * array_in_true * w_p)[0, 0])
    p_SIN_p = np.real((w_p.H * array_sin * w_p)[0, 0])

    SINR_p = np.log10((p_SIN_p - p_IN_p) / p_IN_p) * 10.0
    #print(p_IN_p, p_SIN_p)
    #print(p_IN_r, p_SIN_r)
    total_cnt = total_cnt+1
    print(a, np.mean(np.real(SINR_old)), SINR_p, SINR_r)
    if(SINR_p>np.mean(np.real(SINR_old)) and SINR_p>0):
        gain_cnt +=1
        db_gain += SINR_p
        db_gain -= np.mean(np.real(SINR_old))
tick2 = time.time()
print(gain_cnt,'/',total_cnt)
print(db_gain/gain_cnt)
print('time',tick2-tick)




