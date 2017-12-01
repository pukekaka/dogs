from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
# from gensim.models.doc2vec import LabeledSentence
import smart_open
import gensim

import time
import os
import logging
import time
import collections
import random
from sklearn.decomposition import PCA
from matplotlib import pyplot
from collections import namedtuple
from collections import OrderedDict
import numpy as np

from pprint import pprint

from os import listdir
from os.path import isfile, join

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

current_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = 'output'
data_directory2 = 'each'
filename = 'basicblock_by_file'
model_directory = 'model'
modelname = filename+'_doc2vec.model'
file_path = os.path.join(current_directory, data_directory, data_directory2)
test_file_path = os.path.join(current_directory, data_directory)
model_path = os.path.join(current_directory, model_directory, modelname)

hashLabels = []
hashLabels = [f for f in listdir(file_path)]

print(hashLabels)

inst_list = []
for h in hashLabels:
    buf = ''
    f = open(file_path + '/' + h, 'r')
    for line in f.readlines():
        buf = buf + line.strip() + ' '
    inst_list.append(buf)

print(inst_list)
print(len(hashLabels))
print(len(inst_list))

class HashIterator(object):
    def __init__(self, inst_list, hashLabels):
        self.hashLabels = hashLabels
        self.inst_list = inst_list
    def __iter__(self):
        for idx, inst in enumerate(self.inst_list):
            # print(hashLabels[idx], '---', inst)
            yield TaggedDocument(inst.split(), [self.hashLabels[idx]])

iterator = HashIterator(inst_list, hashLabels)

model = doc2vec.Doc2Vec(size=300, window=10, min_count=1, workers=11,alpha=0.025, min_alpha=0.025) # use fixed learning rate
model.build_vocab(iterator)

epoch = 10
model.alpha -= 0.002 # decrease the learning rate
model.min_alpha = model.alpha # fix the learning rate, no deca
model.train(iterator, total_examples=model.corpus_count, epochs=epoch)

model.save(model_path)
print(model)
print(model.most_similar('pushzero'))
print(model.docvecs.most_similar('abexcm1.exe_bb_by_line'))
# print(model['pushzero'])



#
#
#
#
#
# data = []
# for doc in docLabels:
#     buf = ''
#     f = open(file_path + '/' + doc, 'r')
#     for line in f.readlines():
#         buf = buf + line.strip() + ' '
#     data.append(buf)
#
# print(len(data))
#
# class DocIterator(object):
#     def __init__(self, doc_list, labels_list):
#         self.labels_list = labels_list
#         self.doc_list = doc_list
#
#     def __iter__(self):
#         for idx, doc in enumerate(self.doc_list):
#             yield TaggedDocument(words=doc.split(), labels=[self.labels_list[idx]])




# def read_corpus(fname):
#     for doc in docLabels:
#         path = os.path.join(fname, doc)
#         with smart_open.smart_open(path, encoding="utf8") as f:
#             buf = ''
#             for line in enumerate(f):
#                 buf = buf + str(line)
#             yield TaggedDocument(gensim.utils.simple_preprocess(buf.strip(), min_len=1, max_len=100), [doc])
#
# train_corpus = list(read_corpus(file_path))
#
# model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)
# model.build_vocab(train_corpus)




#
# print(train_corpus)

# for doc in docLabels:
#     # buf = ''
#     buf = []
#     path = os.path.join(file_path, doc)
#     f = open(path, 'r')
#     for line in f.readlines():
#         buf = buf + line.split()
#     f.close()
#     print('done reading', doc)
#     data.append(buf)
#     # for fp in listdir(path):
#     #     print(fp)
#         # data.append(open(file_path + '/' + doc, 'r'))

# data = []
# for doc in docLabels:
#     buf = ''
#     path = os.path.join(file_path, doc)
#     f = open(path, 'r')
#     for line in f.readlines():
#         buf += line.strip()
#     f.close()
#     print('done reading', doc)
#     data.append(buf)


# data = []
# for doc in docLabels:
#     path = os.path.join(file_path, doc)
#     data.append(open(path, 'r'))

# class DocIterator(object):
#     def __init__(self, doc_list, labels_list):
#         self.labels_list = labels_list
#         self.doc_list = doc_list
#
#     def __iter__(self):
#         for idx, doc in enumerate(self.doc_list):
#             print(doc)
#             yield TaggedDocument(words=doc.split(), labels=[self.labels_list[idx]])
#
#
#
#
# model = doc2vec.Doc2Vec(size=100, window=10, min_count=1, workers=4, alpha=0.025, min_alpha=0.025)
# # model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)
# model.build_vocab(iterator)
#
# print('done building vocabulary')
# print('start training the model')
#
# for epoch in range(10):
#     print('epcho %d..' % (epoch+1))
#     model.train(iterator)
#     model.alpha -= 0.002
#     model.min_alpha = model.alpha
#     model.train(iterator)
#
# model.save(model_path)


#
#
# def read_corpus(fname, tokens_only=False):
#     with smart_open.smart_open(fname, encoding="utf8") as f:
#         for i, line in enumerate(f):
#             # print(type(line))
#             if tokens_only:
#                 yield gensim.utils.simple_preprocess(line)
#             else:
#                 yield TaggedDocument(gensim.utils.simple_preprocess(line.strip(), min_len=1, max_len=100), [i])
#
# train_corpus = list(read_corpus(file_path))
#
# # print(train_corpus)
#
# model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)
# model.build_vocab(train_corpus)
#
# # %time model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
# # print(model.infer_vector(['pushzero']))
#

#
# insts_list = [[insts for insts in insts_line.lower().split()] for insts_line in insts_corpus]
# model = word2vec.Word2Vec(insts_list, min_count=1, size=200, workers=4)
#
# model.save(model_path)
#
# result = model.most_similar(positive=['deceax'], topn=5)
# for el in result:
#     print (el)
#
# X = model[model.wv.vocab]
#
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
#
# pyplot.scatter(result[:, 0], result[:, 1])
# instructions = list(model.wv.vocab)
#
# for i, instruction in enumerate(instructions):
#     pyplot.annotate(instruction, xy=(result[i, 0], result[i, 1]))
#
# pyplot.show()
#
#
# # new_model = word2vec.Word2Vec.load(model_path)
# # print(new_model)
#
# end = time.time() - start
#
# print('time', end)
