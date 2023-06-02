# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:19:47 2020

@author: 王非翊
"""

import numpy as np
import random
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="4,5"
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.8)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))  

from tensorflow.keras import backend, callbacks, optimizers #,losses

import urnet_inv_gru4
from tensorflow.keras import backend as K 
K.set_image_data_format('channels_last') 
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
def seed_tensorflow(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
seed_tensorflow(42)


input_nx = 168
input_nz = 1
output_nz = 1500
dataset_numbers = 100000
input_name = 'noise_sample_zrf100000_168_1.rf_sum'
output_name1 = 'vp_100000_1500_1.bin'
output_name2 = 'vs_100000_1500_1.bin'
output_name3 = 'h_100000_600_1.bin'
output_name4 = 'dis_vp_100000_200_1.bin'
output_name5 = 'dis_vs_100000_200_1.bin'

filepath='invert.hdf5'
epochs = 1

batchsize = 64

#读入3维二进制数据
#读入float型数据
def get_float_bin_data3D(filename, nx, ny, nz):
    pic = np.fromfile(filename,dtype=np.float32)
    pic = pic.reshape(nx,ny,nz)
    return pic
#读入2维二进制数据
#读入float型数据
def get_float_bin_data2D(filename, nx, ny):
    pic = np.fromfile(filename,dtype=np.float32)
    pic = pic.reshape(nx,ny)
    return pic

# [2000, 1, 800]
input_data= get_float_bin_data3D(input_name,dataset_numbers,input_nz,input_nx)

output_data1=get_float_bin_data2D(output_name1,dataset_numbers,output_nz)
output_data2=get_float_bin_data2D(output_name2,dataset_numbers,output_nz)
output_data3=get_float_bin_data2D(output_name3,dataset_numbers,600)
output_data4=get_float_bin_data2D(output_name4,dataset_numbers,200)
output_data5=get_float_bin_data2D(output_name5,dataset_numbers,200)



(nz,ny,nx)=input_data.shape
(nz1,ny1)=output_data1.shape
(nz2,ny2)=output_data2.shape
(nz3,ny3)=output_data3.shape
(nz4,ny4)=output_data4.shape
(nz5,ny5)=output_data5.shape
#print("output_data2.shape",output_data2.shape)






# [2000*800, 800, 1]
# [1, 800]
train_input = np.reshape(input_data,(len(input_data),nx, ny))
train_output1 = output_data1
train_output2 = output_data2
train_output3 = output_data3
train_output4 = output_data4
train_output5 = output_data5




#check
train_input=train_input[0:32,:,:]
output_data1=train_output1[0:32,:]
output_data2=train_output2[0:32,:]
output_data3=train_output3[0:32,:]
output_data4=train_output4[0:32,:]
output_data5=train_output5[0:32,:]

'''
#plot dis
x=np.arange(0,200)

trend_pred=np.reshape(dis_vp,200)*5000

trend_pred=trend_pred[0:200]
#print(trend_pred)

plt.plot(x,trend_pred)

plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)

plt.show()
'''
#print(train_input.shape)
#print(train_output.shape)
#check


#indices = np.random.permutation(train_input0.shape[0])
#rand_input = train_input[indices]
#rand_output1 = output_data1[indices]
#rand_output2 = output_data2[indices]




backend.clear_session()


input_size = (nx, ny)

def get_lr_metric(optimizer):  # printing the value of the learning rate
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
#输入层
model = urnet_inv_gru4.s2vpnet16(num_classes=output_nz)
model.build(input_shape=(None,nx, ny))
optimizer=optimizers.Adam(lr=0.001)
lr = get_lr_metric(optimizer)
model.compile(optimizer,loss={'output_1':'mse','output_2':'mse',
                              'output_3':'mse','output_4':'mse','output_5':'mse'},
              metrics=[lr],loss_weights={'output_1':0.3, 'output_2':0.2,
                                         'output_3': 0.1,'output_4':0.2,'output_5':0.2})
model.summary()

callbacks_list = [callbacks.ModelCheckpoint(filepath,monitor='val_loss',
                                            verbose=0,
                                            save_best_only=True,mode='auto',
                                            save_weights_only=True),
                  callbacks.ReduceLROnPlateau(monitor='val_loss',
                                              factor=0.5, patience=3,
                                              mode='min',min_delta=1e-8,
                                              min_lr = 0.0000000001)]


history = model.fit(train_input, (output_data1,output_data2,
                                  output_data3,output_data4,output_data5),
                    validation_split=0.2,
                    epochs=epochs, batch_size=batchsize,verbose = 2,
                    callbacks=callbacks_list)



