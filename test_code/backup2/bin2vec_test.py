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
model_path = os.path.join(current_directory, model_directory, modelname)

docLabels = []
docLabels = [f for f in listdir(file_path)]

def read_corpus(fname):
    for doc in docLabels:
        path = os.path.join(fname, doc)
        with smart_open.smart_open(path, encoding="utf8") as f:
            buf = ''
            for line in enumerate(f):
                buf = buf + str(line)
            yield TaggedDocument(gensim.utils.simple_preprocess(buf.strip(), min_len=1, max_len=100), [doc])

train_corpus = list(read_corpus(file_path))

model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)
model.build_vocab(train_corpus)

# print(model.infer_vector(['pushzero']))

# ranks = []
# second_ranks = []
# for doc_id in range(len(train_corpus)):
#     inferred_vector = model.infer_vector(train_corpus[doc_id].words)
#     sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
#     rank = [docid for docid, sim in sims].index(doc_id)
#     ranks.append(rank)
#
#     second_ranks.append(sims[1])
#
# collections.Counter(ranks)
# print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
# print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
# for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
#     print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
#
# doc_id = random.randint(0, len(train_corpus))
#
# # Compare and print the most/median/least similar documents from the train corpus
# print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
# sim_id = second_ranks[doc_id]
# print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))


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
# iterator = DocIterator(data, docLabels)
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
# ranks = []
# second_ranks = []
# for doc_id in range(len(train_corpus)):
#     inferred_vector = model.infer_vector(train_corpus[doc_id].words)
#     sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
#     rank = [docid for docid, sim in sims].index(doc_id)
#     ranks.append(rank)
#
#     second_ranks.append(sims[1])
#
# collections.Counter(ranks)
# print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
# print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
# for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
#     print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
#
# doc_id = random.randint(0, len(train_corpus))
#
# # Compare and print the most/median/least similar documents from the train corpus
# print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
# sim_id = second_ranks[doc_id]
# print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))

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
