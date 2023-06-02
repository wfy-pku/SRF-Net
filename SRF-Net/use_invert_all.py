# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:19:47 2020

@author: 王非翊
"""
import numpy as np
# from tensorflow.keras.applications import ResNet50
import urnet_inv_gru6
import matplotlib.pyplot as plt
input_nx = 168
input_nz = 1
output_nz = 1500

test_nam = 'test_1000'
test_name = test_nam+'.srf'


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


model = urnet_inv_gru6.s2vpnet16(num_classes=output_nz)
model.build(input_shape=(None,input_nx, input_nz))
model.summary()

model.load_weights('invert666.hdf5')

test_input=get_float_bin_data3D(test_name,1000,input_nz,input_nx)
#test_input=test_input[11:12,:,:]
(nz,ny,nx)=test_input.shape
test = np.reshape(test_input,(len(test_input),nx, ny))
#test_input = np.reshape(test_data, (len(test_data),1, 32,32))  
#print(test)

#(vp,vs,h,dis_vp,dis_vs,srf)= model.predict(test)
(vp,vs,h,dis_vp,dis_vs)= model.predict(test)

file = open(test_nam+".vp","wb")
#file.write(struct.pack('f',pred.shape))
file.write(vp)
file.close()

file = open(test_nam+".vs","wb")
#file.write(struct.pack('f',pred.shape))
file.write(vs)
file.close()


file = open(test_nam+".h","wb")
#file.write(struct.pack('f',pred.shape))
file.write(h)
file.close()

file = open(test_nam+".dis_vp","wb")
#file.write(struct.pack('f',pred.shape))
file.write(dis_vp)
file.close()

file = open(test_nam+".dis_vs","wb")
#file.write(struct.pack('f',pred.shape))
file.write(dis_vs)
file.close()

'''
file = open(test_nam+".denoised_srf","wb")
#file.write(struct.pack('f',pred.shape))
file.write(x6)
file.close()
x
file = open(test_nam+".denoised_srf","wb")
#file.write(struct.pack('f',pred.shape))
file.write(srf)
file.close()

#plot trend
x=np.arange(0,1500)

trend_pred=np.reshape(vp,1500)

trend_pred=trend_pred[0:1500]*5000

plt.plot(x,trend_pred)

plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)

plt.show()

#plot dis
x=np.arange(0,200)

trend_pred=np.reshape(dis_vp,200)*5000

trend_pred=trend_pred[0:200]
#print(trend_pred)

plt.plot(x,trend_pred)

plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)

plt.show()

#plot h
x=np.arange(0,600)

trend_pred=np.reshape(h,600)

trend_pred=trend_pred[0:600]
yy=0
vv=0
for i in range(0, 600, 1):
            if yy <= trend_pred[i]:
                yy=trend_pred[i]
                vv=i
h_dl=vv-50+250
print("h_dl=%f\n"%h_dl) 
plt.plot(x,trend_pred,label="h dl=%.3f"%h_dl)

plt.plot(x,trend_pred)

plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)

plt.show()

#plot srf
x=np.arange(0,128)

trend_pred=np.reshape(srf,168)

trend_pred=trend_pred[0:128]
print(trend_pred)

plt.plot(x,trend_pred)

plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)

plt.show()

'''



