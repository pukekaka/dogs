import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import csv
import string
import requests
import io
from zipfile import ZipFile
from tensorflow.contrib import learn
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

save_file_name = os.path.join('temp', 'temp_spam_data.csv')

if not os.path.exists('temp'):
    os.makedirs('temp')

if os.path.isfile(save_file_name):
    text_data = []
    with open(save_file_name, 'r') as temp_output_file:
        reader = csv.reader(temp_output_file)
        for row in reader:
            text_data.append(row)

else:
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')

    text_data = file.decode()
    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode().split('\n')
    text_data = [x.split('\t') for x in text_data if len(x) >= 1]

    with open(save_file_name, 'w') as temp_output_file:
        writer = csv.writer(temp_output_file)
        writer.writerows(text_data)

#
# for x in text_data:
#     print(len(x))

texts = [x[1] for x in text_data if len(x) > 0]
target = [x[0] for x in text_data if len(x) > 0]
target = [1 if x == 'spam' else 0 for x in target]

# print(texts)

texts = [x.lower() for x in texts]
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
texts = [' '.join(x.split()) for x in texts]

# print(texts)

text_lengths = [len(x.split()) for x in texts]
# print(max(text_lengths))
text_lengths = [x for x in text_lengths if x < 50]
# print(max(text_lengths))
plt.hist(text_lengths, bins=25)
# plt.hist(text_lengths)
plt.title('Histogram of # of Words in Texts')
# plt.show()

sentence_size = 25
min_word_freq = 3

vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size, min_frequency=min_word_freq)
vocab_processor.fit_transform(texts)
embedding_size = len(vocab_processor.vocabulary_)

train_indices = np.random.choice(len(texts), round(len(texts)*0.8), replace=False)
test_indices = np.array(list(set(range(len(texts))) - set(train_indices)))
texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
target_train = [x for ix, x in enumerate(target)]
target_test = [x for ix, x in enumerate(target) if ix in test_indices]

identity_mat = tf.diag(tf.ones(shape=[embedding_size]))

A = tf.Variable(tf.random_normal(shape=[embedding_size, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

x_data = tf.placeholder(shape=[sentence_size], dtype=tf.int32)
y_target = tf.placeholder(shape=[1, 1], dtype=tf.float32)

x_embed = tf.nn.embedding_lookup(identity_mat, x_data)
x_col_sums = tf.reduce_sum(x_embed, 0)

x_col_sums_2D = tf.expand_dims(x_col_sums, 0)
print(x_col_sums)
print(x_col_sums_2D)
print(A)
model_output = tf.add(tf.matmul(x_col_sums_2D, A), b)


loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
prediction = tf.sigmoid(model_output)
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

loss_vec = []
train_acc_all = []
train_acc_avg = []
for ix, t in enumerate(vocab_processor.fit_transform(texts_train)):
    y_data = [[target_train[ix]]]

    sess.run(train_step, feed_dict={x_data: t, y_target: y_data})
    temp_loss = sess.run(loss, feed_dict={x_data: t, y_target: y_data})
    loss_vec.append(temp_loss)

    if (ix + 1) % 10 == 0:
        print('Training Observation #' + str(ix + 1) + ': Loss = ' + str(temp_loss))

    # Keep trailing average of past 50 observations accuracy
    # Get prediction of single observation
    [[temp_pred]] = sess.run(prediction, feed_dict={x_data: t, y_target: y_data})
    # Get True/False if prediction is accurate
    train_acc_temp = target_train[ix] == np.round(temp_pred)
    train_acc_all.append(train_acc_temp)
    if len(train_acc_all) >= 50:
        train_acc_avg.append(np.mean(train_acc_all[-50:]))

# # Get test set accuracy
# print('Getting Test Set Accuracy For {} Sentences.'.format(len(texts_test)))
# test_acc_all = []
# for ix, t in enumerate(vocab_processor.fit_transform(texts_test)):
#     y_data = [[target_test[ix]]]
#
#     if (ix + 1) % 50 == 0:
#         print('Test Observation #' + str(ix + 1))
#
#         # Keep trailing average of past 50 observations accuracy
#     # Get prediction of single observation
#     [[temp_pred]] = sess.run(prediction, feed_dict={x_data: t, y_target: y_data})
#     # Get True/False if prediction is accurate
#     test_acc_temp = target_test[ix] == np.round(temp_pred)
#     test_acc_all.append(test_acc_temp)
#
# print('\nOverall Test Accuracy: {}'.format(np.mean(test_acc_all)))
#
# # Plot training accuracy over time
# plt.plot(range(len(train_acc_avg)), train_acc_avg, 'k-', label='Train Accuracy')
# plt.title('Avg Training Acc Over Past 50 Generations')
# plt.xlabel('Generation')
# plt.ylabel('Training Accuracy')
# plt.show()