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

# docLabels = []
# docLabels = [f for f in listdir(file_path)]

SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')

alldocs = []  # Will hold all docs in original order
with open(test_file_path + '/alldata-id.txt', encoding='utf-8') as alldata:
    for line_no, line in enumerate(alldata):
        tokens = gensim.utils.to_unicode(line).split()
        words = tokens[1:]
        tags = [line_no] # 'tags = [tokens[0]]' would also work at extra memory cost
        split = ['train', 'test', 'extra', 'extra'][line_no//25000]  # 25k train, 25k test, 25k extra
        sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//12500] # [12.5K pos, 12.5K neg]*2 then unknown
        alldocs.append(SentimentDocument(words, tags, split, sentiment))

train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']
doc_list = alldocs[:]  # For reshuffling per pass

print('%d docs: %d train-sentiment, %d test-sentiment' % (len(doc_list), len(train_docs), len(test_docs)))

#
#
# cores = multiprocessing.cpu_count()
# assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"
#
# simple_models = [
#     # PV-DM w/ concatenation - window=5 (both sides) approximates paper's 10-word total window size
#     Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=cores),
#     # PV-DBOW
#     Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores),
#     # PV-DM w/ average
#     Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=cores),
# ]
#
# # Speed up setup by sharing results of the 1st model's vocabulary scan
# simple_models[0].build_vocab(alldocs)  # PV-DM w/ concat requires one special NULL word so it serves as template
# print(simple_models[0])
# for model in simple_models[1:]:
#     model.reset_from(simple_models[0])
#     print(model)
#
# models_by_name = OrderedDict((str(model), model) for model in simple_models)
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
#
#
# iterator = DocIterator(data, docLabels)
#
# model = doc2vec.Doc2Vec(size=300, window=10, min_count=1, workers=11,alpha=0.025, min_alpha=0.025) # use fixed learning rate
# model.build_vocab(iterator)
# for epoch in range(10):
#     model.train(iterator)
#     model.alpha -= 0.002 # decrease the learning rate
#     model.min_alpha = model.alpha # fix the learning rate, no deca
#     model.train(iterator)


# train_corpus = "toy_data/train_docs.txt"
# docs = g.doc2vec.TaggedLineDocument(train_corpus)
# model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, pretrained_emb=pretrained_emb, iter=train_epoch)
#
# #save model
# model.save(saved_path)


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
