# Import TensorFlow and enable eager execution
# This code requires TensorFlow version >=1.9
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
from utilis import *
import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

# -------import pakage----------------
# feel free to change these parameters according to your system's configuration

def get_args():
    """ Defines training-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Attention-based image caption model')
    parser.add_argument('--BATCH_SIZE', default=20, help='batch size of the dataset')

    # Add data arguments
    parser.add_argument('--BUFFER_SIZE', default=1000, help='buffer size')
    parser.add_argument('--embedding_dim', default=256, help='source language')
    parser.add_argument('--units', default=512, help='target language')
    parser.add_argument('--EPOCHS', default= 40, type= int, help='train model on a tiny dataset')

    # Add model arguments
    parser.add_argument('--score_model', default="default", help='score of attention:default,general,concat')

    # Add optimization arguments
    parser.add_argument('--models', default="GRU", help='force stop training at specified epoch')
    parser.add_argument('--local_att', default="False", help='clip threshold of gradients')
    # Add checkpoint arguments
   # parser.add_argument('--log-file', default=None, help='path to save logs')
    parser.add_argument('--file_pre', default='prediction.txt', help='path to save checkpoints')
    parser.add_argument('--file_tar', default='target.txt', help='filename to load checkpoint')
    parser.add_argument('--file_losstrain', default= "losstrainresult.txt", help='save a checkpoint every N epochs')
    parser.add_argument('--file_lossval', default='lossvalresult.txt', help='don\'t save models or checkpoints')
    parser.add_argument('--attention_features_shape', default=100, help='don\'t save models or checkpoints')
    
    parser.add_argument('--DDD', default=1, help='target language')
    
  #  parser.add_argument('--context_vector_concat', default=True, help='target language')
    parser.add_argument('--num_layers', default=1, help='target language')
    parser.add_argument('--context_vector_concat', default="True", help='target language')
    # Parse twice as model arguments are not known the first time
    args = parser.parse_args()
    #ARCH_CONFIG_REGISTRY[args.arch](args)
    return args


def main(args):
    BATCH_SIZE = args.BATCH_SIZE
    BUFFER_SIZE = args.BUFFER_SIZE
    embedding_dim = args.embedding_dim
    DDD_t = args.DDD
    context_vector_concat_t = args.context_vector_concat
    units = args.units
    # shape of the vector extracted from InceptionV3 is (64, 2048)
    # these two variables represent that
    features_shape = 2048
    attention_features_shape = args.attention_features_shape
    EPOCHS = args.EPOCHS
    score_model="concat"
    #"default":
    # "general":
    # "concat"
    num_layers= args.num_layers
    context_vector_concat=args.context_vector_concat
    models=args.models
    local_att=args.local_att
    file_pre = open(str('./lossdir/' + args.file_pre),'w')
    file_tar = open(str('./lossdir/' + args.file_tar),'w')
    file_losstrain = open(str('./lossdir/'+args.file_losstrain),'w')
    file_lossval = open(str('./lossdir/' + args.file_lossval),'w')
    

    #---------------------hyoer-param

    socket.setdefaulttimeout(20)
    annotation_zip = tf.keras.utils.get_file('captions.zip', 
                                              cache_subdir=os.path.abspath('.'),
                                              origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                              extract = True)
    annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'

    name_of_zip = 'train2014.zip'
    if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
        image_zip = tf.keras.utils.get_file(name_of_zip,cache_subdir=os.path.abspath('.'),
                                               origin = 'http://images.cocodataset.org/zips/train2014.zip',
                                               extract = True)

        PATH = os.path.dirname(image_zip)+'/train2014/'
    else:
        PATH = os.path.abspath('.')+'/train2014/'


    #---------------data-------------------------
    # read the json file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # storing the captions and the image name in vectors
    all_captions = []
    all_img_name_vector = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)
        # all_img_name_vector 是所有照片地址
        all_img_name_vector.append(full_coco_image_path)
        # all_captions 是所有照片的caption
        all_captions.append(caption)

    # shuffling the captions and image_names together
    # setting a random state
    #将照片和照片地址打包并随机打乱
    train_captions, img_name_vector = shuffle(all_captions,
                                              all_img_name_vector,
                                              random_state=1)

    # selecting the first 30000 captions from the shuffled set
    #取前30000个照片出来做 train
    #######################################
    #######################################
    num_examples = 30000
    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]
    print("dataset_size")
    #---------------choose_the_dataset_size------------------


    #使用xception来做预处理，resize image 为了能够塞入 pretrained models中 之前使用inceptionv3
    image_model = tf.keras.applications.xception.Xception(include_top=False, 
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    # getting the unique images
    encode_train = sorted(set(img_name_vector))

    # feel free to change the batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(
                                     encode_train).map(load_image).batch(16)

    #用Dataset创建一个image_dataset包含已经处理过的image和地址 并分成了batch_size = 16. 
    #for (img, path) in image_dataset:
     #    batch_features = image_features_extract_model(img)
      #   batch_features = tf.reshape(batch_features, 
       #                            (batch_features.shape[0], -1, batch_features.shape[3]))
#
 #        for bf, p in zip(batch_features, path):
  #           path_of_feature = p.numpy().decode("utf-8")
   #          np.save(path_of_feature, bf.numpy())
    #将batch_features提取出来 并和 path一起保存起来

    #-------------------------cnngetthefeature------------------------
    # This will find the maximum length of any caption in our dataset

    # The steps above is a general process of dealing with text processing
    # choosing the top 5000 words from the vocabulary
    #############################
    #############################
    top_k = 10000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k, 
                                                      oov_token="<unk>", 
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    tokenizer.word_index = {key:value for key, value in tokenizer.word_index.items() if value <= top_k}
    # putting <unk> token in the word2idx dictionary
    tokenizer.word_index[tokenizer.oov_token] = top_k + 1
    tokenizer.word_index['<pad>'] = 0

    # creating the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    # creating a reverse mapping (index -> word)
    index_word = {value:key for key, value in tokenizer.word_index.items()}
    vocab_size = len(tokenizer.word_index)
    index_word[1]='<unk>'

    # padding each vector to the max_length of the captions
    # if the max_length parameter is not provided, pad_sequences calculates that automatically
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    #--------------------- text---------------

    img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector, 
                                                                        cap_vector, 
                                                                        test_size=0.2, 
                                                                        random_state=0)

    train_length=len(cap_train)
    val_length=len(cap_val)

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
    dataset_val = tf.data.Dataset.from_tensor_slices((img_name_val, cap_val))
    # using map to load the numpy files in parallel
    # NOTE: Be sure to set num_parallel_calls to the number of CPU cores you have
    # https://www.tensorflow.org/api_docs/python/tf/py_func
    dataset = dataset.map(lambda item1, item2: tf.py_func(
              map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=8)
    dataset_val = dataset_val.map(lambda item1, item2: tf.py_func(
              map_func, [item1, item2], [tf.float32, tf.int32]), num_parallel_calls=8)
    # shuffling and batching
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset_val = dataset_val.shuffle(BUFFER_SIZE)
    # https://www.tensorflow.org/api_docs/python/tf/contrib/data/batch_and_drop_remainder
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)
    dataset_val = dataset_val.batch(BATCH_SIZE)
    dataset_val = dataset_val.prefetch(1)
    print("dataset finished")
#-------------------dataset------------------------

    encoder = CNN_Encoder(embedding_dim)
    #embedding_dim, units, vocab_size,

    decoder = RNN_Decoder(embedding_dim, units, vocab_size, directional=False, dropout=0, models=models, \
                          num_layers=num_layers, attention_feature_shape = attention_features_shape)

    optimizer = tf.train.AdamOptimizer()
    print("Begin training !")
    # We are masking the loss calculated for padding

    result_ulti=[]
    target_ulti = []
    loss_plot = []
    loss_plot1 = []

    for epoch in range(EPOCHS):
        start = time.time()
        total_loss = 0
        total_valloss=0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            start_1 = time.time()
            print(batch,"-----------------------")
            loss = 0
            # initializing the hidden state for each batch
            # because the captions are not related from image to image
            hidden = decoder.reset_state(batch_size=target.shape[0])

            dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

            with tf.GradientTape() as tape:
                features = encoder(img_tensor)
                for i in range(1, target.shape[1]):
                    # passing the features through the decoder
                    predictions, hidden, _ = decoder(dec_input, features, hidden, score_model,DDD_t,context_vector_concat_t,local_att)
                    loss += loss_function(target[:, i], predictions)

                    # using teacher forcing
                    dec_input = tf.expand_dims(target[:, i], 1)

            total_loss += (loss / int(target.shape[1]))

            variables = encoder.variables + decoder.variables

            gradients = tape.gradient(loss, variables) 

            optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())
            print(batch,"---------------",loss,"-------------",time.time()-start_1,"-------------------")
            if batch % 100 == 0:
                print ('Epoch {} Batch {} Train_Loss {:.4f}'.format(epoch + 1, 
                                                              batch, 
                                                              loss.numpy() / int(target.shape[1])))

        # storing the epoch end loss value to plot later
        loss_plot.append((total_loss / train_length).numpy())
        if epoch == EPOCHS-1:
            file_losstrain.write(str(loss_plot))
            file_losstrain.close()

        print ('Epoch {} Train_Loss {:.6f}'.format(epoch + 1, 
                                             total_loss/train_length))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    #     #------------------------validation---------------------
        start_val=time.time()
        for (batch, (img_tensor, target)) in enumerate(dataset_val):
            loss_val = 0
            result=[]
            target_list= []
            #print(target.shape)
            # initializing the hidden state for each batch
            # because the captions are not related from image to image
            hidden = decoder.reset_state(batch_size=target.shape[0])

            dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)


            features = encoder(img_tensor)
            prediction_length=0
            for i in range(1, target.shape[1]):
                prediction_length +=1
                # passing the features through the decod
                predictions, hidden, _ = decoder(dec_input, features, hidden, score_model, DDD_t, context_vector_concat_t,local_att)
                predicted_id = tf.multinomial(predictions, num_samples=1)
               # print("predictionid",predicted_id,target[:, i] )
                prediction_li=predicted_id.numpy().tolist()
                if epoch == EPOCHS-1:
                    for k in prediction_li:
                        result.append(index_word[k[0]])

                loss_val += loss_function(target[:, i], predictions)
                if epoch == EPOCHS-1:   
                    target_li=target[:, i].numpy().tolist()
                    for k in target_li:
                        target_list.append(index_word[k])

                # using teacher forcing
                dec_input = predicted_id
            total_valloss += (loss_val / prediction_length)
            if epoch == EPOCHS-1:
                result_ulti.append(result)
                target_ulti.append(target_list)
            print(batch,"-----------------------------------------------")
            if batch % 100 == 0:
                print ('Epoch {} Batch {} Validation_Loss {:.4f}'.format(epoch + 1, 
                                                              batch, 
                                                              loss_val.numpy() / prediction_length))

        # storing the epoch end loss value to plot later
        loss_plot1.append((total_valloss / val_length).numpy())

        print ('Epoch {} Validation_Loss {:.6f}'.format(epoch + 1, 
                                             total_valloss/val_length))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start_val))

        if epoch == EPOCHS-1:
            for temp in result_ulti:
                tempnum= np.array(temp).reshape(48,20).T
                for i in range(tempnum.shape[0]):
                    count = 0
                    for j in tempnum[i,:]:
                        count += 1
                        if j == '<end>':
                            file_pre.write('.')
                            file_pre.write('\n')
                            break
                        elif count ==48:
                            file_pre.write('.')
                            file_pre.write('\n')
                        else :
                            file_pre.write(j+' ')
            file_pre.close()

        if epoch == EPOCHS-1:
            for temp in target_ulti:
                tempnum= np.array(temp).reshape(48,20).T
                for i in range(tempnum.shape[0]):
                    count = 0
                    for j in tempnum[i,:]:
                        count += 1
                        if j == '<end>':
                            file_tar.write('.')
                            file_tar.write('\n')
                            break
                        else :
                            file_tar.write(j+' ')
            file_tar.close()
            file_lossval.write(str(loss_plot1))
            file_lossval.close()


    #------------------------------------------

    plt.plot(loss_plot1,label="loss of val")
    plt.plot(loss_plot,label="loss of train")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.legend()
    plt.show()
    #-------------------------------------------


if __name__ == '__main__':
    args = get_args()
    main(args)

