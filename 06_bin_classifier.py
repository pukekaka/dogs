import tensorflow as tf

import os
import numpy as np
from test_code.mann import util
from test_code.mann import model
from test_code.mann import param

iv = param.init_value()
mann = model.memory_augmented_neural_networks(iv)

current_directory = os.path.dirname(os.path.abspath(__file__))
model_directory = 'model'
bin2vec_directory = 'bin_model'
files_path = os.path.join(current_directory, model_directory, bin2vec_directory)

data_loader = util.SampleDataLoader(
            model_dir=files_path,
            insts_size=iv.insts_size,
            n_train_classes=iv.n_train_classes,
            n_test_classes=iv.n_test_classes
        )


def test_f(args, y, output):
    correct = [0] * args.seq_length
    total = [0] * args.seq_length
    y_decode = util.one_hot_decode(y)
    output_decode = util.one_hot_decode(output)
    for i in range(np.shape(y)[0]):
        y_i = y_decode[i]
        output_i = output_decode[i]
        # print(y_i)
        # print(output_i)
        class_count = {}
        for j in range(args.seq_length):
            if y_i[j] not in class_count:
                class_count[y_i[j]] = 0
            class_count[y_i[j]] += 1
            total[class_count[y_i[j]]] += 1
            if y_i[j] == output_i[j]:
                correct[class_count[y_i[j]]] += 1
    return [float(correct[i]) / total[i] if total[i] > 0. else 0. for i in range(1, 11)]




with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())
    tf.global_variables_initializer().run()
    train_writer = tf.summary.FileWriter(iv.tensorboard_dir+'/'+'mann', sess.graph)
    print(iv)
    print("1st\t2nd\t3rd\t4th\t5th\t6th\t7th\t8th\t9th\t10th\tbatch\tloss")

    for b in range(iv.num_epoches):

        # Result Test
        if b % 100 == 0:
            x_inst, x_label, y = data_loader.fetch_batch(iv.n_classes, iv.batch_size, iv.seq_length,
                                                          type='test')
            feed_dict = {mann.x_inst: x_inst, mann.x_label: x_label, mann.y: y}
            output, learning_loss = sess.run([mann.o, mann.learning_loss], feed_dict=feed_dict)
            merged_summary = sess.run(mann.learning_loss_summary, feed_dict=feed_dict)
            train_writer.add_summary(merged_summary, b)
            accuracy = test_f(iv, y, output)
            for accu in accuracy:
                print('%.4f' % accu, end='\t')
            print('%d\t%.4f' % (b, learning_loss))

        # Saving
        if b % 5000 == 0 and b > 0:
            saver.save(sess, iv.model_dir + '/' + 'mann.tfmodel', global_step=b)

        # Training
        x_inst, x_label, y = data_loader.fetch_batch(iv.n_classes, iv.batch_size, iv.seq_length, type='train')
        feed_dict = {mann.x_inst: x_inst, mann.x_label: x_label, mann.y: y}
        sess.run(mann.train_op, feed_dict=feed_dict)
        # print('Step', b)