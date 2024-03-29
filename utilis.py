import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

# We'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import backend as K
#from tensorflow.keras.engine.topology import Layer
from tensorflow.python.keras.layers import Layer, InputSpec
import numpy as np

import re
import numpy as np
import os
import time
import json
from glob import glob
#from PIL import Image
#from PIL import Image
import pickle
import socket


def load_image(image_path):
    img = tf.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_images(img, (299, 299))
    img = tf.keras.applications.xception.preprocess_input(img)
    return img, image_path

def calc_max_length(tensor):
    return max(len(t) for t in tensor)

def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap

def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)


def gru( units,dropout, directional):
  # If you have a GPU, we recommend using the CuDNNGRU layer (it provides a 
  # significant speedup).
    if directional== False:       
        if tf.test.is_gpu_available():
            return tf.keras.layers.CuDNNGRU(units,
                                        return_sequences=True, 
                                        return_state=True, 
                                        recurrent_initializer='glorot_uniform')
        else:
            return tf.keras.layers.GRU(units, dropout= dropout,
                                   return_sequences=True, 
                                   return_state=True, 
                                   recurrent_activation='sigmoid', 
                                   recurrent_initializer='glorot_uniform')
    else:
        if tf.test.is_gpu_available():
            return tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNGRU(units, dropout= dropout,
                                        return_sequences=True, 
                                        return_state=True, 
                                        recurrent_initializer='glorot_uniform'), merge_mode='concat')
        else:
            return tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units, dropout= dropout,
                                   return_sequences=True, 
                                   return_state=True, 
                                   recurrent_activation='sigmoid', 
                                   recurrent_initializer='glorot_uniform'),  merge_mode='concat')
        
        
        

def lstm( units, dropout, directional ):
    if tf.test.is_gpu_available():
        if directional:
            return tf.keras.layers.Bidirectional( keras.layers.CuDNNLSTM(units,return_sequences=True, 
                                    return_state=True, 
                                    recurrent_initializer='glorot_uniform' ), merge_mode='concat')
        else: 

            return tf.keras.layers.CuDNNLSTM(units, 
                            kernel_initializer='glorot_uniform',
                            return_sequences=True, return_state=True )
    else:
        if directional:
            return tf.keras.layers.Bidirectional( keras.layers.LSTM(units,return_sequences=True, dropout= dropout,
                                    return_state=True,  kernel_initializer='glorot_uniform',
                                    recurrent_initializer='glorot_uniform' ), merge_mode='concat')                
                
        else:

            return tf.keras.layers.LSTM(units,return_sequences=True, 
                                    return_state=True, 
                                    recurrent_initializer='glorot_uniform' )
        
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units, attention_feature_shape, local_att,D):
        super(BahdanauAttention, self).__init__()
        self.attention_feature_shape = attention_feature_shape
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.V1 = tf.keras.layers.Dense(1)
        self.V3 = tf.keras.layers.Dense(units)
        if local_att =='True':
            self.V2 =tf.keras.layers.Dense(5)
        else:
            self.V2 = tf.keras.layers.Dense(attention_feature_shape)
        self.W3 = tf.keras.layers.Dense(units)
        self.local_att_layer_1 = local_att_layer(2,D)
        #self.local_att_layer_2 = local_att_layer(attention_feature_shape)

    def call(self, features, hidden, score_model,D=1,context_vector_concat = True,local_att = "False" ):
        #print(local_att)
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        # score shape == (batch_size, 64, hidden_size)
        if local_att =="False":
            if score_model== "default":
                score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
            elif score_model =="general":
                score = tf.matmul( hidden_with_time_axis,tf.nn.tanh(tf.transpose(self.W1(features),[0,2,1])) )

            elif score_model== "concat":
                temp_v = tf.transpose (self.V2( tf.transpose( tf.concat( [ hidden_with_time_axis, self.W1(features)], axis=-2 ), [0, 2,1]) )\
                                       ,[0,2,1])
                score = self.V3( tf.nn.tanh(temp_v)  )

            # attention_weights shape == (batch_size, 64, 1)
            # we get 1 at the last axis because we are applying score to self.V

            attention_weights = tf.cast(tf.nn.softmax(self.V(score), axis=1),tf.float32)
        #if local attention is called to implement in the expriment, the technique is from papre: 
        # Effective Approaches to Attention-based Neural Machine Translation
        batch_size = tf.shape(features).numpy()[0]
        features_shape = tf.shape(features).numpy()[2]
        for i in range(batch_size):
            for j in range(2 * D + 1):
                for k in range(2 * D + 1):
                    if j == 0 and k == 0:
                        temp = tf.reshape(
                            np.linalg.norm(np.array([j, k]) - np.array([(2 * D + 1) / 2, (2 * D + 1) / 2])), (1, 1, 1))
                    else:
                        temp = tf.concat(
                            [temp,
                             tf.reshape(np.linalg.norm(np.array([j, k]) - np.array([(2 * D + 1) / 2, (2 * D + 1) / 2])),
                                        (1, 1, 1))],
                            axis=1)
            if i == 1:
                temp_1 = tf.concat([temp, temp], axis=0)
            elif i>1:
                temp_1 = tf.concat([temp_1, temp], axis=0)

        if local_att=="True":
            #print("1",local_att)
            self.local_att_layer_1.build(hidden_with_time_axis.get_shape().as_list())
            #self.local_att_layer_2.build(hidden_with_time_axis.get_shape().as_list())
            with tf.device('/gpu:0'):
                p_t_1 =  self.local_att_layer_1(hidden_with_time_axis, self.attention_feature_shape)
                features_att = tf.reshape(features, (batch_size,10,10,features_shape))
                for num_batch in range(batch_size):
                    x = tf.slice(p_t_1, [num_batch, 0, 0], [1,1,1]).numpy()[0][0][0]
                    y = tf.slice(p_t_1, [num_batch, 0, 1], [1,1,1]).numpy()[0][0][0]
                    if num_batch == 0:
                        local_feature = tf.slice(features_att,[num_batch, tf.cast(x-D,tf.int32),tf.cast( y-D,tf.int32), 0],[1, 2*D+1,2*D+1, features_shape] )
                    else:
                        local_feature = tf.concat([local_feature, tf.slice(features_att,[num_batch, tf.cast(x-D,tf.int32), tf.cast(y-D,tf.int32), 0],[1, (2*D+1), (2*D+1),features_shape] )], axis=0 )
                local_feature = tf.reshape(local_feature, (batch_size, tf.square(tf.cast( (2*D+1),tf.float32)), features_shape ) )

                if score_model== "default":
                    score_local = tf.nn.tanh(self.W1(local_feature) + self.W2(hidden_with_time_axis))

                elif score_model =="general":
                    score_local = tf.matmul( hidden_with_time_axis,tf.nn.tanh(tf.transpose(self.W1(local_feature),[0,2,1])) )

                #elif score_model== "concat":
                #    temp_v = tf.transpose (self.V2( tf.transpose( tf.concat( [ hidden_with_time_axis, self.W1(local_feature)], axis=-2 ), [0, 2,1]) )\
                #                           ,[0,2,1])
                #    score_local = self.V3( tf.nn.tanh(temp_v)  )

                # attention_weights shape == (batch_size, 64, 1)
                # we get 1 at the last axis because we are applying score to self.V

                attention_weights = tf.cast(tf.nn.softmax(self.V1(score_local), axis=1),tf.float32)
                #print('attention_Weight', attention_weights.shape)
                #print('temp_1', temp_1.shape)

            attention_weights = attention_weights * tf.math.exp(- tf.div( tf.cast(tf.square(temp_1),tf.float32), 0.5 *tf.square(tf.cast(D, tf.float32))) )
            features = local_feature
        # context_vector shape after sum == (batch_size, hidden_size)

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        if context_vector_concat == "True":
            #print(context_vector_concat)
            context_vector_with_hidden = self.W3(tf.nn.tanh( tf.concat( [context_vector, hidden], axis=-1 ) ) )
        else:
            #print(context_vector_concat)
            context_vector_with_hidden = context_vector
        #context_vector_with_hidden = self.W3(tf.nn.tanh( tf.concat( [context_vector, hidden], axis=-1 ) ) )
        return context_vector_with_hidden, attention_weights
    
    
    
    
class local_att_layer(Layer):
    """
    credit to author from: https://www.jianshu.com/p/6c34045216fb 
    The function of this layer is to construct a class of a defined layer
    """

    def __init__(self, output_dim, D, **kwargs):
        self.output_dim = output_dim
        self.D = D
        super(local_att_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel_a = self.add_weight(name='kernel', 
                                      shape=(input_shape[-1], 100),   #!!!!!
                                      initializer='uniform',
                                      trainable=True)
        self.kernel_b = self.add_weight(name='kerne2', 
                                      shape=(100, self.output_dim),     #!!!!!!!
                                      initializer='uniform',
                                      trainable=True)
        super(local_att_layer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, s):
        temp_t = K.dot(tf.tanh(K.dot(x, self.kernel_a)), self.kernel_b)
        temp_t = tf.nn.sigmoid(temp_t)* (tf.sqrt(tf.cast(s, tf.float32))- 2*self.D) + self.D
        return temp_t


    
    
    
class CNN_Encoder(tf.keras.Model):
    # Since we have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)
        
    def call(self, x):      
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

    
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size,
                 models, num_layers , attention_feature_shape,local_att_t,D=1, directional=False, dropout=0):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.dropout = dropout
        self.model = models
        self.directional= directional
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn = tf.keras.models.Sequential()
        if self.model =="GRU":
            for i in range(num_layers):
#                 self.rnn.add(tf.keras.layers.GRU(self.units, dropout= 0,
#                                    return_sequences=True, 
#                                    return_state=True, 
#                                    recurrent_activation='sigmoid', 
#                                    recurrent_initializer='glorot_uniform'))
                self.rnn.add(gru( self.units, dropout= 0, directional=self.directional ))


        elif self.model =="LSTM":
            for i in range(num_layers):
                self.rnn.add(lstm( self.units, self.dropout, self.directional ))
        #elif self.model =="":
        self.W1 = tf.keras.layers.Dense(self.units)
        self.W2 = tf.keras.layers.Dense(self.units)
        self.W3 = tf.keras.layers.Dense(self.units)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.directional= directional

        self.attention = BahdanauAttention(self.units, attention_feature_shape, local_att_t, D)

    def call(self, x, features, hidden,score_model, ddd,context_vector_concat_t,local_att_t):
        #print(self.model)
        # defining attention as a separate model
        if self.model =="LSTM":
            #print(self.model,1)
            hidden_state = hidden[0]
        else:
            hidden_state = hidden
            #features, hidden, score_model,D=1,context_vector_concat = True,local_att = True
        context_vector_with_hidden, attention_weights = self.attention(features, hidden_state,score_model,
                                                                       D=ddd, context_vector_concat=context_vector_concat_t,local_att=local_att_t )

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        dropout_1 = tf.keras.layers.Dropout(0.5)
        dropout_2 = tf.keras.layers.Dropout(0.5)
        dropout_3 = tf.keras.layers.Dropout(0.5)
        x = self.embedding(x)
        x = dropout_1(x)
        #x_t = x
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size*2)
        if self.model =="LSTM":
            #print(self.model,2)
            cell = hidden[1]
            context_vector_with_hidden = tf.concat([context_vector_with_hidden, cell], axis=-1 )
        x = tf.concat([tf.expand_dims(context_vector_with_hidden, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        if self.model == "LSTM":
            output, state, cell = self.rnn(x)
            state = [state, cell]
#         else:
#             state = [state, cell]

        elif self.model =="GRU":
            output, state = self.rnn(x)
        # shape == (batch_size, max_length, hidden_size)

        x = self.fc1(output)
        x = dropout_2(x)
        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = dropout_3(x)
        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        if self.model=="GRU":
            return tf.zeros((batch_size, self.units))
        if self.model == "LSTM":
            list_state_cell= [tf.zeros((batch_size, self.units  )),tf.zeros((batch_size, self.units  ))] 
            return list_state_cell

#----------------------method and function
