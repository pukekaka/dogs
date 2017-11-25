from gensim.models import word2vec

import time
import os
import logging
from sklearn.decomposition import PCA
from matplotlib import pyplot

from pprint import pprint

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

start = time.time()

current_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = 'output'
filename = 'basicblock_by_line'
model_directory = 'model'
modelname = filename+'-word2vec.model'
file_path = os.path.join(current_directory, data_directory , filename)
model_path = os.path.join(current_directory, model_directory, modelname)

insts_corpus = list()

fr = open(file_path, 'r', encoding='utf-8')

while True:
    line = fr.readline()
    if not line:
        break
    insts_corpus.append(line.strip())

fr.close()

insts_list = [[insts for insts in insts_line.lower().split()] for insts_line in insts_corpus]
model = word2vec.Word2Vec(insts_list, min_count=1, size=200, workers=4)

model.save(model_path)

result = model.most_similar(positive=['deceax'], topn=5)
for el in result:
    print (el)

X = model[model.wv.vocab]

pca = PCA(n_components=2)
result = pca.fit_transform(X)

pyplot.scatter(result[:, 0], result[:, 1])
instructions = list(model.wv.vocab)

for i, instruction in enumerate(instructions):
    pyplot.annotate(instruction, xy=(result[i, 0], result[i, 1]))

pyplot.show()


# new_model = word2vec.Word2Vec.load(model_path)
# print(new_model)

end = time.time() - start

print('time', end)
