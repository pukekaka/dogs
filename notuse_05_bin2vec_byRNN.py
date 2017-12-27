from gensim.models import doc2vec
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

current_directory = os.path.dirname(os.path.abspath(__file__))
filename = 'basicblock_by_file'
model_directory = 'model'
model_directory2 = 'func_model'
modelname = filename+'_func2vec.model'
modellistname = 'func2vec_model_list'
model_path = os.path.join(current_directory, model_directory, model_directory2, modelname)
model_list_path = os.path.join(current_directory, model_directory, modellistname)



# funcvec = model.docvecs['fff564d59deec80ad5fcc92867e07b69_17']
# sims = model.docvecs.most_similar('02456317d10e2d769f704935d5b5b6cd_2137')
# sims2 = model.most_similar('intthree')
# print(funcvec)
# print(sims2)


# '''
# Load model
# '''
# model = doc2vec.Doc2Vec.load(model_path)
#
#
# '''
# Setting hash-functionLabel list
# '''
# f = open(model_list_path, 'r')
# lines = f.readlines()
# f.close()
#
# hash_num_list = list()
# for line in lines:
#     hash_num = line.split()[0]
#     hash_num_list.append(hash_num.split('_'))
# # print(hash_num_list)
#
# hash_list = list(set(hash_num[0] for hash_num in hash_num_list))
# # print(len(hash_list))
#
# hash_funclist_dict = dict()
# hash_list_size = len(hash_list)
# for idx, h in enumerate(hash_list):
#     funclist = []
#     for hash_num in hash_num_list:
#         if h == hash_num[0]:
#             funclist.append(str(hash_num[0]+'_'+hash_num[1]))
#     hash_funclist_dict[h] = funclist
#     print(idx, '/', hash_list_size, h, 'make hash_funclist_dict completed')
# # print(len(hash_funclist_dict['1ee939f18be02962d8406d2aa640b294']))
#
#
# '''
# Setting hash-functionVector list
# '''
# hfdk_size = len(hash_funclist_dict.keys())
# hash_funcveclist_dict = dict()
# for idx, hash_key in enumerate((hash_funclist_dict.keys())):
#     func_idx_list = hash_funclist_dict[hash_key]
#     func_vec_list = list()
#     for func_idx in func_idx_list:
#         func_vec = model.docvecs[func_idx]
#         func_vec_list.append(func_vec)
#     hash_funcveclist_dict[hash_key] = func_vec_list
#     print(idx, '/', hfdk_size, hash_key, 'make hash_funcveclist_dict')
#
# # print(len(hash_funcveclist_dict['1fc456ca4363d041f9557463157daabe']))
#
#
# '''
# LSTM hash-functionVector list
# '''
#
# learning_rate = 0.001
# total_epoch = 30
# batch_size = 128
#
# data_dim = 50
# # seqlen = tf.placeholder(tf.int32, [None])
# seq_length = []
# for hash_funcvec_key in hash_funcveclist_dict.keys():
#     seq_length.append(len(hash_funcveclist_dict[hash_funcvec_key]))
# output_dim = 1
# n_hidden = 128
# max_value = max(seq_length)

# n_step = tf.placeholder(tf.int32)

# X = tf.placeholder(tf.float32, [None, max_value, data_dim])
# Y = tf.placeholder(tf.float32, [None, 1])

# lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
# output, states = tf.nn.dynamic_rnn(lstm_cell, X, sequence_length=seq_length, dtype=tf.float32)
# Y_last = tf.contrib.layers.fully_connected(output[:, -1], output_dim, activation_fn=None)





def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# train Parameters
seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 500

# Open, High, Low, Volume, Close
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
print(xy)
print(xy.shape)
# xy = xy[::-1]  # reverse order (chronically ordered)
# xy = MinMaxScaler(xy)
# x = xy
# y = xy[:, [-1]]  # Close as label
#
# # build a dataset
# dataX = []
# dataY = []
# for i in range(0, len(y) - seq_length):
#     _x = x[i:i + seq_length]
#     _y = y[i + seq_length]  # Next close price
#     # print(_x, "->", _y)
#     dataX.append(_x)
#     dataY.append(_y)
#
# # train/test split
# train_size = int(len(dataY) * 0.7)
# test_size = len(dataY) - train_size
# trainX, testX = np.array(dataX[0:train_size]), np.array(
#     dataX[train_size:len(dataX)])
# trainY, testY = np.array(dataY[0:train_size]), np.array(
#     dataY[train_size:len(dataY)])
#
# # input place holders
# X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
# Y = tf.placeholder(tf.float32, [None, 1])
#
# # build a LSTM network
# cell = tf.contrib.rnn.BasicLSTMCell(
#     num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
# outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
# Y_pred = tf.contrib.layers.fully_connected(
#     outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output
#
# # cost/loss
# loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# # optimizer
# optimizer = tf.train.AdamOptimizer(learning_rate)
# train = optimizer.minimize(loss)
#
# # RMSE
# targets = tf.placeholder(tf.float32, [None, 1])
# predictions = tf.placeholder(tf.float32, [None, 1])
# rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
#
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#
#     # Training step
#     for i in range(iterations):
#         _, step_loss = sess.run([train, loss], feed_dict={
#                                 X: trainX, Y: trainY})
#         print("[step: {}] loss: {}".format(i, step_loss))
#
#     # Test step
#     test_predict = sess.run(Y_pred, feed_dict={X: testX})
#     rmse_val = sess.run(rmse, feed_dict={
#                     targets: testY, predictions: test_predict})
#     print("RMSE: {}".format(rmse_val))
#
#     # Plot predictions
#     plt.plot(testY)
#     plt.plot(test_predict)
#     plt.xlabel("Time Period")
#     plt.ylabel("Stock Price")
#     plt.show()