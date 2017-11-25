from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
import smart_open
import gensim

import time
import os
import logging
import time
from sklearn.decomposition import PCA
from matplotlib import pyplot


from pprint import pprint

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

start = time.time()

current_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = 'output'
filename = 'basicblock_by_line'
model_directory = 'model'
modelname = filename+'-phrase2vec.model'
file_path = os.path.join(current_directory, data_directory , filename)
model_path = os.path.join(current_directory, model_directory, modelname)

def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="utf8") as f:
        for i, line in enumerate(f):
            # print(type(line))
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                yield TaggedDocument(gensim.utils.simple_preprocess(line.strip(), min_len=1, max_len=100), [i])
                # yield TaggedDocument(), [i])
                # result = list(gensim.utils.tokenize(line.strip(), encoding='utf8', lowercase=False))
                # result = [inst for inst in line]
                # result = list(line)
                # print(line)
                # print(list(line))
                # print(result)
                # yield 'test'

train_corpus = list(read_corpus(file_path))

# print(train_corpus)

model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)
model.build_vocab(train_corpus)

# %time model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
print(model.infer_vector(['pushzero']))




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
