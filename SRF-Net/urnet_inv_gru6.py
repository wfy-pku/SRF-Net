import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.regularizers import l2
import keras.backend as K
from keras.layers.recurrent import GRU


class S2vpBlock1(layers.Layer):

    def __init__(self, filter_num, strides=1):
        super(S2vpBlock1, self).__init__()

        self.conv1 = layers.Conv1D(filter_num, strides=strides,kernel_size=7, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv1D(filter_num,strides=1 ,kernel_size=5, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')
       # self.maxp = layers.MaxPooling1D(pool_size=2)
        self.bn2 = layers.BatchNormalization()
        if strides != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv1D(filter_num,1, strides=strides))
        else:
            self.downsample = lambda x:x


    def call(self, inputs, training=None):

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)
        output = layers.add([out, identity])
      #  output = self.maxp(output)
        output = tf.nn.relu(output)
        return output


class S2vpNet(keras.Model):

    def __init__(self, num_classes):  # [2, 2, 2, 2]
        super(S2vpNet, self).__init__()
#        self.bnin = layers.BatchNormalization()
        k=48
        
        self.maxpool_spatial = layers.Lambda(lambda x: K.max(x, axis=2, keepdims=True))
        self.avgpool_spatial = layers.Lambda(lambda x: K.mean(x, axis=2, keepdims=True))
        self.max_avg_pool_spatial = layers.Concatenate(axis=2)
        self.convsp = layers.Conv1D(filters=1, kernel_size=7, padding="same", activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        self.mult2 = layers.Multiply()
        
        self.gru1 = layers.GRU(512, input_shape = (128,1))
        self.gdp1 = layers.Dropout(0.0)
        
        
        
        self.concsrf = layers.Concatenate(axis=1)
        self.stem = Sequential([layers.Conv1D(k, kernel_size=1, strides=1),
                                layers.BatchNormalization(),
                                layers.Activation('relu')])

        self.layer1 = self.build_s2vpblock1(k,blocks=1)
        self.layer2 = self.build_s2vpblock1(k*2,blocks=1,strides=2)
        self.layer3 = self.build_s2vpblock1(k*4,blocks=1,strides=2)
        self.layer4 = self.build_s2vpblock1(k*8,blocks=1,strides=2)
        self.layer5 = self.build_s2vpblock1(k*16,blocks=1,strides=2)
        
#Vp             
#1#################################################

        self.Us0 = layers.UpSampling1D(size = (2))
        self.conv0 = layers.Conv1D(k*8, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')
        self.conc0 = layers.Concatenate(axis=1)
        self.bn0 = layers.BatchNormalization()
        self.downsample0 = lambda x:x
        self.uss0 = layers.UpSampling1D(size=(4))
        self.conv00 = layers.Conv1D(k*8, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.bn00 = layers.BatchNormalization()
#2#################################################

        self.Us1 = layers.UpSampling1D(size = (2))
        self.conv1 = layers.Conv1D(k*4, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')
        self.conc1 = layers.Concatenate(axis=1)
        self.bn1 = layers.BatchNormalization()
        self.downsample1 = lambda x:x
        self.uss1 = layers.UpSampling1D(size=(3))
        self.conv11 = layers.Conv1D(k*4, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.bn11 = layers.BatchNormalization()
#3#################################################

        self.Us2 = layers.UpSampling1D(size = (2))
        self.conv2 = layers.Conv1D(k*2, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')
        self.conc2 = layers.Concatenate(axis=1)
        self.bn2 = layers.BatchNormalization()
        self.downsample2 = lambda x:x
        self.uss2 = layers.UpSampling1D(size=(3))
        self.conv22 = layers.Conv1D(k*2, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.bn22 = layers.BatchNormalization()
        
#4#################################################

        self.Us3 = layers.UpSampling1D(size = (2))
        self.conv3 = layers.Conv1D(k, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')
        self.conc3 = layers.Concatenate(axis=1)
        self.bn3 = layers.BatchNormalization()
        self.downsample3 = lambda x:x
        self.uss3 = layers.UpSampling1D(size=(3))
        self.conv33 = layers.Conv1D(k, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.bn33 = layers.BatchNormalization()
        

        
#5#################################################
        self.conv4 = layers.Conv1D(1, kernel_size=3, padding='same', 
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')   
        
        self.bn4 = layers.BatchNormalization()

        self.fat1 = layers.Flatten()
        self.fc1 = layers.Dense(units=1024,activation='relu')
        self.dp1 = layers.Dropout(0.3)
        self.fc2 = layers.Dense(units=1024,activation='relu')
        self.dp2 = layers.Dropout(0.3)
        self.fc3 = layers.Dense(units=num_classes,activation='sigmoid')

#vs        
#1#################################################

        self.vs_Us0 = layers.UpSampling1D(size = (2))
        self.vs_conv0 = layers.Conv1D(k*8, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')
        self.vs_conc0 = layers.Concatenate(axis=1)
        self.vs_bn0 = layers.BatchNormalization()
        self.vs_downsample0 = lambda x:x
        self.vs_uss0 = layers.UpSampling1D(size=(4))
        self.vs_conv00 = layers.Conv1D(k*8, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.vs_bn00 = layers.BatchNormalization()
#2#################################################

        self.vs_Us1 = layers.UpSampling1D(size = (2))
        self.vs_conv1 = layers.Conv1D(k*4, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')
        self.vs_conc1 = layers.Concatenate(axis=1)
        self.vs_bn1 = layers.BatchNormalization()
        self.vs_downsample1 = lambda x:x
        self.vs_uss1 = layers.UpSampling1D(size=(3))
        self.vs_conv11 = layers.Conv1D(k*4, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.vs_bn11 = layers.BatchNormalization()
#3#################################################

        self.vs_Us2 = layers.UpSampling1D(size = (2))
        self.vs_conv2 = layers.Conv1D(k*2, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')
        self.vs_conc2 = layers.Concatenate(axis=1)
        self.vs_bn2 = layers.BatchNormalization()
        self.vs_downsample2 = lambda x:x
        self.vs_uss2 = layers.UpSampling1D(size=(3))
        self.vs_conv22 = layers.Conv1D(k*2, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.vs_bn22 = layers.BatchNormalization()
        
#4#################################################

        self.vs_Us3 = layers.UpSampling1D(size = (2))
        self.vs_conv3 = layers.Conv1D(k, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')
        self.vs_conc3 = layers.Concatenate(axis=1)
        self.vs_bn3 = layers.BatchNormalization()
        self.vs_downsample3 = lambda x:x
        self.vs_uss3 = layers.UpSampling1D(size=(3))
        self.vs_conv33 = layers.Conv1D(k, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.vs_bn33 = layers.BatchNormalization()
        

        
#5#################################################
        self.vs_conv4 = layers.Conv1D(1, kernel_size=3, padding='same', 
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')   
        
        self.vs_bn4 = layers.BatchNormalization()

        self.vs_fat1 = layers.Flatten()
        self.vs_fc1 = layers.Dense(units=1024,activation='relu')
        self.vs_dp1 = layers.Dropout(0.3)
        self.vs_fc2 = layers.Dense(units=1024,activation='relu')
        self.vs_dp2 = layers.Dropout(0.3)
        self.vs_fc3 = layers.Dense(units=num_classes,activation='sigmoid') 
        
##h
#1#################################################

        self.h_Us0 = layers.UpSampling1D(size = (2))
        self.h_conv0 = layers.Conv1D(k*8, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-4),
                                   activation='relu')
        self.h_conc0 = layers.Concatenate(axis=1)
        self.h_bn0 = layers.BatchNormalization()
        self.h_downsample0 = lambda x:x
        self.h_uss0 = layers.UpSampling1D(size=(4))
        self.h_conv00 = layers.Conv1D(k*8, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.h_bn00 = layers.BatchNormalization()
#2#################################################

        self.h_Us1 = layers.UpSampling1D(size = (2))
        self.h_conv1 = layers.Conv1D(k*4, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-4),
                                   activation='relu')
        self.h_conc1 = layers.Concatenate(axis=1)
        self.h_bn1 = layers.BatchNormalization()
        self.h_downsample1 = lambda x:x
        self.h_uss1 = layers.UpSampling1D(size=(3))
        self.h_conv11 = layers.Conv1D(k*4, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.h_bn11 = layers.BatchNormalization()
#3#################################################

        self.h_Us2 = layers.UpSampling1D(size = (2))
        self.h_conv2 = layers.Conv1D(k*2, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-4),
                                   activation='relu')
        self.h_conc2 = layers.Concatenate(axis=1)
        self.h_bn2 = layers.BatchNormalization()
        self.h_downsample2 = lambda x:x
        self.h_uss2 = layers.UpSampling1D(size=(3))
        self.h_conv22 = layers.Conv1D(k*2, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.h_bn22 = layers.BatchNormalization()
        
#4#################################################

        self.h_Us3 = layers.UpSampling1D(size = (2))
        self.h_conv3 = layers.Conv1D(k, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-4),
                                   activation='relu')
        self.h_conc3 = layers.Concatenate(axis=1)
        self.h_bn3 = layers.BatchNormalization()
        self.h_downsample3 = lambda x:x
        self.h_uss3 = layers.UpSampling1D(size=(3))
        self.h_conv33 = layers.Conv1D(k, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.h_bn33 = layers.BatchNormalization()
        

        
#5#################################################
        self.h_conv4 = layers.Conv1D(1, kernel_size=3, padding='same', 
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-4),
                                   activation='relu')   
        
        self.h_bn4 = layers.BatchNormalization()

        self.h_fat1 = layers.Flatten()
        self.h_fc1 = layers.Dense(units=1024,activation='relu')
        self.h_dp1 = layers.Dropout(0.5)
        self.h_fc2 = layers.Dense(units=1024,activation='relu')
        self.h_dp2 = layers.Dropout(0.5)
        self.h_fc3 = layers.Dense(units=600,activation='sigmoid')

#dis_vp        
#1#################################################

        self.dis_Us0 = layers.UpSampling1D(size = (2))
        self.dis_conv0 = layers.Conv1D(k*8, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')
        self.dis_conc0 = layers.Concatenate(axis=1)
        self.dis_bn0 = layers.BatchNormalization()
        self.dis_downsample0 = lambda x:x
        self.dis_uss0 = layers.UpSampling1D(size=(4))
        self.dis_conv00 = layers.Conv1D(k*8, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.dis_bn00 = layers.BatchNormalization()
#2#################################################

        self.dis_Us1 = layers.UpSampling1D(size = (2))
        self.dis_conv1 = layers.Conv1D(k*4, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')
        self.dis_conc1 = layers.Concatenate(axis=1)
        self.dis_bn1 = layers.BatchNormalization()
        self.dis_downsample1 = lambda x:x
        self.dis_uss1 = layers.UpSampling1D(size=(3))
        self.dis_conv11 = layers.Conv1D(k*4, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.dis_bn11 = layers.BatchNormalization()
#3#################################################

        self.dis_Us2 = layers.UpSampling1D(size = (2))
        self.dis_conv2 = layers.Conv1D(k*2, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')
        self.dis_conc2 = layers.Concatenate(axis=1)
        self.dis_bn2 = layers.BatchNormalization()
        self.dis_downsample2 = lambda x:x
        self.dis_uss2 = layers.UpSampling1D(size=(3))
        self.dis_conv22 = layers.Conv1D(k*2, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.dis_bn22 = layers.BatchNormalization()
        
#4#################################################

        self.dis_Us3 = layers.UpSampling1D(size = (2))
        self.dis_conv3 = layers.Conv1D(k, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')
        self.dis_conc3 = layers.Concatenate(axis=1)
        self.dis_bn3 = layers.BatchNormalization()
        self.dis_downsample3 = lambda x:x
        self.dis_uss3 = layers.UpSampling1D(size=(3))
        self.dis_conv33 = layers.Conv1D(k, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.dis_bn33 = layers.BatchNormalization()
        

        
#5#################################################
        self.dis_conv4 = layers.Conv1D(1, kernel_size=3, padding='same', 
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')   
        
        self.dis_bn4 = layers.BatchNormalization()

        self.dis_fat1 = layers.Flatten()
        self.dis_fc1 = layers.Dense(units=1024,activation='relu')
        self.dis_dp1 = layers.Dropout(0.3)
        self.dis_fc2 = layers.Dense(units=1024,activation='relu')
        self.dis_dp2 = layers.Dropout(0.3)
        self.dis_fc3 = layers.Dense(units=200,activation='sigmoid')
        
#dis_vs       
#1#################################################

        self.dis_vs_Us0 = layers.UpSampling1D(size = (2))
        self.dis_vs_conv0 = layers.Conv1D(k*8, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')
        self.dis_vs_conc0 = layers.Concatenate(axis=1)
        self.dis_vs_bn0 = layers.BatchNormalization()
        self.dis_vs_downsample0 = lambda x:x
        self.dis_vs_uss0 = layers.UpSampling1D(size=(4))
        self.dis_vs_conv00 = layers.Conv1D(k*8, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.dis_vs_bn00 = layers.BatchNormalization()
#2#################################################

        self.dis_vs_Us1 = layers.UpSampling1D(size = (2))
        self.dis_vs_conv1 = layers.Conv1D(k*4, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')
        self.dis_vs_conc1 = layers.Concatenate(axis=1)
        self.dis_vs_bn1 = layers.BatchNormalization()
        self.dis_vs_downsample1 = lambda x:x
        self.dis_vs_uss1 = layers.UpSampling1D(size=(3))
        self.dis_vs_conv11 = layers.Conv1D(k*4, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.dis_vs_bn11 = layers.BatchNormalization()
#3#################################################

        self.dis_vs_Us2 = layers.UpSampling1D(size = (2))
        self.dis_vs_conv2 = layers.Conv1D(k*2, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')
        self.dis_vs_conc2 = layers.Concatenate(axis=1)
        self.dis_vs_bn2 = layers.BatchNormalization()
        self.dis_vs_downsample2 = lambda x:x
        self.dis_vs_uss2 = layers.UpSampling1D(size=(3))
        self.dis_vs_conv22 = layers.Conv1D(k*2, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.dis_vs_bn22 = layers.BatchNormalization()
        
#4#################################################

        self.dis_vs_Us3 = layers.UpSampling1D(size = (2))
        self.dis_vs_conv3 = layers.Conv1D(k, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')
        self.dis_vs_conc3 = layers.Concatenate(axis=1)
        self.dis_vs_bn3 = layers.BatchNormalization()
        self.dis_vs_downsample3 = lambda x:x
        self.dis_vs_uss3 = layers.UpSampling1D(size=(3))
        self.dis_vs_conv33 = layers.Conv1D(k, kernel_size=1,padding='same',
                                    kernel_initializer='he_normal')
        self.dis_vs_bn33 = layers.BatchNormalization()
        

        
#5#################################################
        self.dis_vs_conv4 = layers.Conv1D(1, kernel_size=3, padding='same', 
                                   kernel_initializer='he_normal', 
                                   kernel_regularizer=l2(1e-6),
                                   activation='relu')   
        
        self.dis_vs_bn4 = layers.BatchNormalization()

        self.dis_vs_fat1 = layers.Flatten()
        self.dis_vs_fc1 = layers.Dense(units=1024,activation='relu')
        self.dis_vs_dp1 = layers.Dropout(0.3)
        self.dis_vs_fc2 = layers.Dense(units=1024,activation='relu')
        self.dis_vs_dp2 = layers.Dropout(0.3)
        self.dis_vs_fc3 = layers.Dense(units=200,activation='sigmoid')


        

    def call(self, inputs, training=None):
        
        x_rf=inputs[:,0:128,:]
        print("x_rf.shape",x_rf.shape)
        x_rf1 = self.maxpool_spatial(x_rf)
        x_rf2 = self.avgpool_spatial(x_rf)
        print("x_rf111111.shape",x_rf1.shape)
        print("x_rf222222.shape",x_rf2.shape)
        
        x_rf3 = self.max_avg_pool_spatial([x_rf1,x_rf2])
        x_rf3 = self.convsp(x_rf3)
        x_rf = self.mult2([x_rf3,x_rf])
        
        
        x_rf = self.gru1(x_rf)
        print("x_rfgru.shape",x_rf.shape)
        x_rf = self.gdp1(x_rf)

        x_rf = tf.expand_dims(x_rf, -1)
#        print("x_rf.shape",x_rf.shape)
        x_sf=inputs[:,128:168,:]
        x1 = self.stem(x_sf)
        print("inputs.shape",inputs.shape)
        x1 = self.layer1(x1)     #16
        print("x1.shape",x1.shape)
        x2 = self.layer2(x1)     #32
        print("x2.shape",x2.shape)        
        x3 = self.layer3(x2)     #64
        print("x3.shape",x3.shape)
        x4 = self.layer4(x3)     #128
        print("x4.shape",x4.shape)
        x5 = self.layer5(x4)     #128
        print("x5.shape",x5.shape)
#vp        

        
        x7 = self.Us0(x5)        #64
        x7 = self.conv0(x7)      #64
        x7 = self.conc0([x7, x4])#64
        x7 = self.bn0(x7)        #64
        print("x7.shape",x7.shape)
        identity = self.downsample0(x5)
        identity = self.uss0(identity)  
        print("identity.shape",identity.shape)
        identity = identity[:,0:11,:]
        identity = self.conv00(identity) 
        identity = self.bn00(identity) 
        x7 = layers.add([x7, identity])
        print("x7.shape",x7.shape)

        x8 = self.Us1(x7)       #32
        x8 = self.conv1(x8)   #32
        x8 = self.conc1([x8, x3])#32
        x8 = self.bn1(x8)      #32
        print("x8.shape",x8.shape)
        identity = self.downsample1(x7)
        identity = self.uss1(identity)
        identity = self.conv11(identity)   
        identity = self.bn11(identity) 
        print("identity.shape",identity.shape)
        identity = identity[:,0:32,:]
        x8 = layers.add([x8, identity])
        print("x8.shape",x8.shape)
        
        x9 = self.Us2(x8)       #32
        x9 = self.conv2(x9)   #32      
        x9 = self.conc2([x9, x2])#32
        x9 = self.bn2(x9)
        print("x9.shape",x9.shape)
        identity = self.downsample2(x8)
        identity = self.uss2(identity)
        identity = self.conv22(identity)   
        identity = self.bn22(identity) 
        print("identity.shape",identity.shape)
        identity = identity[:,0:84,:]
        x9 = layers.add([x9, identity])
        print("x9.shape",x9.shape)
        
        x10 = self.Us3(x9)       #32
        x10 = self.conv3(x10)   #32      
        x10 = self.conc3([x10, x1])#32
        x10 = self.bn3(x10)
        print("x10.shape",x10.shape)
        identity = self.downsample3(x9)
        identity = self.uss3(identity)
        identity = self.conv33(identity)   
        identity = self.bn33(identity) 
        print("identity.shape",identity.shape)
        identity = identity[:,0:208,:]
        x10 = layers.add([x10, identity])
        print("x10.shape",x10.shape)
        
        x11 = self.conv4(x10)         
        x12 = self.bn4(x11) 
        print("x12.shape",x12.shape)
        
        x12 = self.concsrf([x12, x_rf])

        x13 = self.fat1(x12)
        x13 = self.fc1(x13)
        x13 = self.dp1(x13)
        x13 = self.fc2(x13)
        x13 = self.dp2(x13)
        x13 = self.fc3(x13)
#Vs
        
        x7_vs = self.vs_Us0(x5)        #64
        x7_vs = self.vs_conv0(x7_vs)      #64
        x7_vs = self.vs_conc0([x7_vs, x4])#64
        x7_vs = self.vs_bn0(x7_vs)        #64
        identity = self.vs_downsample0(x5)
        identity = self.vs_uss0(identity)  
        print("identity.shape",identity.shape)
        identity = identity[:,0:11,:]
        identity = self.vs_conv00(identity) 
        identity = self.vs_bn00(identity) 
        x7_vs = layers.add([x7_vs, identity])
        print("x7.shape",x7_vs.shape)

        x8_vs = self.vs_Us1(x7_vs)       #32
        x8_vs = self.vs_conv1(x8_vs)   #32
        x8_vs = self.vs_conc1([x8_vs, x3])#32
        x8_vs = self.vs_bn1(x8_vs)      #32
        identity = self.vs_downsample1(x7_vs)
        identity = self.vs_uss1(identity)
        identity = self.vs_conv11(identity)   
        identity = self.vs_bn11(identity) 
        identity = identity[:,0:32,:]
        x8_vs = layers.add([x8_vs, identity])
        print("x8.shape",x8_vs.shape)
        
        x9_vs = self.vs_Us2(x8_vs)       #32
        x9_vs = self.vs_conv2(x9_vs)   #32      
        x9_vs = self.vs_conc2([x9_vs, x2])#32
        x9_vs = self.vs_bn2(x9_vs)
        identity = self.vs_downsample2(x8_vs)
        identity = self.vs_uss2(identity)
        identity = self.vs_conv22(identity)   
        identity = self.vs_bn22(identity) 
        identity = identity[:,0:84,:]
        x9_vs = layers.add([x9_vs, identity])
        print("x9.shape",x9.shape)
        
        x10_vs = self.vs_Us3(x9_vs)       #32
        x10_vs = self.vs_conv3(x10_vs)   #32      
        x10_vs = self.vs_conc3([x10_vs, x1])#32
        x10_vs = self.vs_bn3(x10_vs)
        identity = self.vs_downsample3(x9_vs)
        identity = self.vs_uss3(identity)
        identity = self.vs_conv33(identity)   
        identity = self.vs_bn33(identity) 
        identity = identity[:,0:208,:]
        x10_vs = layers.add([x10_vs, identity])
        print("x10.shape",x10_vs.shape)
        
        x11_vs = self.vs_conv4(x10_vs)         
        x12_vs = self.vs_bn4(x11_vs) 
        
        x12_vs = self.concsrf([x12_vs, x_rf])
        
        x13_vs = self.vs_fat1(x12_vs)
        x13_vs = self.vs_fc1(x13_vs)
        x13_vs = self.vs_dp1(x13_vs)
        x13_vs = self.vs_fc2(x13_vs)
        x13_vs = self.vs_dp2(x13_vs)
        x13_vs = self.vs_fc3(x13_vs) 
        print("x13_vs.shape",x13_vs.shape)
        print("x13.shape",x13.shape)
#h        
        x7_h = self.h_Us0(x5)        #64
        x7_h = self.h_conv0(x7_h)      #64
        x7_h = self.h_conc0([x7_h, x4])#64
        x7_h = self.h_bn0(x7_h)        #64
        identity = self.h_downsample0(x5)
        identity = self.h_uss0(identity)  
        print("identity.shape",identity.shape)
        identity = identity[:,0:11,:]
        identity = self.h_conv00(identity) 
        identity = self.h_bn00(identity) 
        x7_h = layers.add([x7_h, identity])
        print("x7.shape",x7_h.shape)

        x8_h = self.h_Us1(x7_h)       #32
        x8_h = self.h_conv1(x8_h)   #32
        x8_h = self.h_conc1([x8_h, x3])#32
        x8_h = self.h_bn1(x8_h)      #32
        identity = self.h_downsample1(x7_h)
        identity = self.h_uss1(identity)
        identity = self.h_conv11(identity)   
        identity = self.h_bn11(identity) 
        identity = identity[:,0:32,:]
        x8_h = layers.add([x8_h, identity])
        print("x8.shape",x8_h.shape)
        
        x9_h = self.h_Us2(x8_h)       #32
        x9_h = self.h_conv2(x9_h)   #32      
        x9_h = self.h_conc2([x9_h, x2])#32
        x9_h = self.h_bn2(x9_h)
        identity = self.h_downsample2(x8_h)
        identity = self.h_uss2(identity)
        identity = self.h_conv22(identity)   
        identity = self.h_bn22(identity) 
        identity = identity[:,0:84,:]
        x9_h = layers.add([x9_h, identity])
        print("x9.shape",x9_h.shape)
        
        x10_h = self.h_Us3(x9_h)       #32
        x10_h = self.h_conv3(x10_h)   #32      
        x10_h = self.h_conc3([x10_h, x1])#32
        x10_h = self.h_bn3(x10_h)
        identity = self.h_downsample3(x9_h)
        identity = self.h_uss3(identity)
        identity = self.h_conv33(identity)   
        identity = self.h_bn33(identity) 
        identity = identity[:,0:208,:]
        x10_h = layers.add([x10_h, identity])
        print("x10.shape",x10_h.shape)
        
        x11_h = self.h_conv4(x10_h)         
        x12_h = self.h_bn4(x11_h) 
        
        x12_h = self.concsrf([x12_h, x_rf])
        
        x13_h = self.h_fat1(x12_h)
        x13_h = self.h_fc1(x13_h)
        x13_h = self.h_dp1(x13_h)
        x13_h = self.h_fc2(x13_h)
        x13_h = self.h_dp2(x13_h)
        x13_h = self.h_fc3(x13_h)
#dis_vp        
 
        x7_dis = self.dis_Us0(x5)        #64
        x7_dis = self.dis_conv0(x7_dis)      #64
        x7_dis = self.dis_conc0([x7_dis, x4])#64
        x7_dis = self.dis_bn0(x7_dis)        #64
        identity = self.dis_downsample0(x5)
        identity = self.dis_uss0(identity)  
        print("identity.shape",identity.shape)
        identity = identity[:,0:11,:]
        identity = self.dis_conv00(identity) 
        identity = self.dis_bn00(identity) 
        x7_dis = layers.add([x7_dis, identity])
        print("x7.shape",x7_dis.shape)

        x8_dis = self.dis_Us1(x7_dis)       #32
        x8_dis = self.dis_conv1(x8_dis)   #32
        x8_dis = self.dis_conc1([x8_dis, x3])#32
        x8_dis = self.dis_bn1(x8_dis)      #32
        identity = self.dis_downsample1(x7_dis)
        identity = self.dis_uss1(identity)
        identity = self.dis_conv11(identity)   
        identity = self.dis_bn11(identity) 
        identity = identity[:,0:32,:]
        x8_dis = layers.add([x8_dis, identity])
        print("x8.shape",x8_dis.shape)
        
        x9_dis = self.dis_Us2(x8_dis)       #32
        x9_dis = self.dis_conv2(x9_dis)   #32      
        x9_dis = self.dis_conc2([x9_dis, x2])#32
        x9_dis = self.dis_bn2(x9_dis)
        identity = self.dis_downsample2(x8_dis)
        identity = self.dis_uss2(identity)
        identity = self.dis_conv22(identity)   
        identity = self.dis_bn22(identity) 
        identity = identity[:,0:84,:]
        x9_dis = layers.add([x9_dis, identity])
        print("x9.shape",x9_dis.shape)
        
        x10_dis = self.dis_Us3(x9_dis)       #32
        x10_dis = self.dis_conv3(x10_dis)   #32      
        x10_dis = self.dis_conc3([x10_dis, x1])#32
        x10_dis = self.dis_bn3(x10_dis)
        identity = self.dis_downsample3(x9_dis)
        identity = self.dis_uss3(identity)
        identity = self.dis_conv33(identity)   
        identity = self.dis_bn33(identity) 
        identity = identity[:,0:208,:]
        x10_dis = layers.add([x10_dis, identity])
        print("x10.shape",x10_dis.shape)
        
        x11_dis = self.dis_conv4(x10_dis)         
        x12_dis = self.dis_bn4(x11_dis) 
        
        x12_dis = self.concsrf([x12_dis, x_rf])

        x13_dis = self.dis_fat1(x12_dis)
        x13_dis = self.dis_fc1(x13_dis)
        x13_dis = self.dis_dp1(x13_dis)
        x13_dis = self.dis_fc2(x13_dis)
        x13_dis = self.dis_dp2(x13_dis)
        x13_dis = self.dis_fc3(x13_dis)
#dis_vs        
 
        x7_dis_vs = self.dis_vs_Us0(x5)        #64
        x7_dis_vs = self.dis_vs_conv0(x7_dis_vs)      #64
        x7_dis_vs = self.dis_vs_conc0([x7_dis_vs, x4])#64
        x7_dis_vs = self.dis_vs_bn0(x7_dis_vs)        #64
        identity = self.dis_vs_downsample0(x5)
        identity = self.dis_vs_uss0(identity)  
        print("identity.shape",identity.shape)
        identity = identity[:,0:11,:]
        identity = self.dis_vs_conv00(identity) 
        identity = self.dis_vs_bn00(identity) 
        x7_dis_vs = layers.add([x7_dis_vs, identity])
        print("x7.shape",x7_dis_vs.shape)

        x8_dis_vs = self.dis_vs_Us1(x7_dis_vs)       #32
        x8_dis_vs = self.dis_vs_conv1(x8_dis_vs)   #32
        x8_dis_vs = self.dis_vs_conc1([x8_dis_vs, x3])#32
        x8_dis_vs = self.dis_vs_bn1(x8_dis_vs)      #32
        identity = self.dis_vs_downsample1(x7_dis_vs)
        identity = self.dis_vs_uss1(identity)
        identity = self.dis_vs_conv11(identity)   
        identity = self.dis_vs_bn11(identity) 
        identity = identity[:,0:32,:]
        x8_dis_vs = layers.add([x8_dis_vs, identity])
        print("x8.shape",x8_dis_vs.shape)
        
        x9_dis_vs = self.dis_vs_Us2(x8_dis_vs)       #32
        x9_dis_vs = self.dis_vs_conv2(x9_dis_vs)   #32      
        x9_dis_vs = self.dis_vs_conc2([x9_dis_vs, x2])#32
        x9_dis_vs = self.dis_vs_bn2(x9_dis_vs)
        identity = self.dis_vs_downsample2(x8_dis_vs)
        identity = self.dis_vs_uss2(identity)
        identity = self.dis_vs_conv22(identity)   
        identity = self.dis_vs_bn22(identity) 
        identity = identity[:,0:84,:]
        x9_dis_vs = layers.add([x9_dis_vs, identity])
        print("x9.shape",x9_dis_vs.shape)
        
        x10_dis_vs = self.dis_vs_Us3(x9_dis_vs)       #32
        x10_dis_vs = self.dis_vs_conv3(x10_dis_vs)   #32      
        x10_dis_vs = self.dis_vs_conc3([x10_dis_vs, x1])#32
        x10_dis_vs = self.dis_vs_bn3(x10_dis_vs)
        identity = self.dis_vs_downsample3(x9_dis_vs)
        identity = self.dis_vs_uss3(identity)
        identity = self.dis_vs_conv33(identity)   
        identity = self.dis_vs_bn33(identity) 
        identity = identity[:,0:208,:]
        x10_dis_vs = layers.add([x10_dis_vs, identity])
        print("x10.shape",x10_dis_vs.shape)
        
        x11_dis_vs = self.dis_vs_conv4(x10_dis_vs)         
        x12_dis_vs = self.dis_vs_bn4(x11_dis_vs) 
        
        
        x12_dis_vs = self.concsrf([x12_dis_vs, x_rf])
        
        
        
        x13_dis_vs = self.dis_vs_fat1(x12_dis_vs)
        x13_dis_vs = self.dis_vs_fc1(x13_dis_vs)
        x13_dis_vs = self.dis_vs_dp1(x13_dis_vs)
        x13_dis_vs = self.dis_vs_fc2(x13_dis_vs)
        x13_dis_vs = self.dis_vs_dp2(x13_dis_vs)
        x13_dis_vs = self.dis_vs_fc3(x13_dis_vs)

        return x13,x13_vs,x13_h,x13_dis,x13_dis_vs

    def build_s2vpblock1(self, filter_num, blocks, strides=1):

        res_blocks = Sequential()
        # may down sample
        res_blocks.add(S2vpBlock1(filter_num, strides))

        for _ in range(1, blocks):
            res_blocks.add(S2vpBlock1(filter_num, strides=1))

        return res_blocks




def s2vpnet16(num_classes):
    return S2vpNet(num_classes=num_classes)
