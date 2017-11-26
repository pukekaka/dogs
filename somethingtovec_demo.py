from gensim.models import word2vec
from gensim.models import doc2vec

import os
import logging
from sklearn.decomposition import PCA
from matplotlib import pyplot

from pprint import pprint

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

current_directory = os.path.dirname(os.path.abspath(__file__))
filename = 'basicblock_by_line'
model_directory = 'model'
modelname = filename+'_inst2vec.model'
dmodelname = filename+'_phrase2vec.model'
model_path = os.path.join(current_directory, model_directory, modelname)
dmodel_path = os.path.join(current_directory, model_directory, dmodelname)

model = word2vec.Word2Vec.load(model_path)
dmodel = doc2vec.Doc2Vec.load(dmodel_path)
print(model)
print(dmodel)

result_wms = model.most_similar(positive=['deceax'], topn=5)
for el in result_wms:
    print('wv', el)

result_dms = dmodel.most_similar(positive=['deceax'], topn=5)
for el in result_dms:
    print('dv', el)


Xw = model[model.wv.vocab]
Xd = dmodel[dmodel.wv.vocab]

pca = PCA(n_components=2)
result_w = pca.fit_transform(Xw)
result_d = pca.fit_transform(Xd)

pyplot.scatter(result_w[:, 0], result_w[:, 1])
# pyplot.scatter(result_d[:, 0], result_d[:, 1])
# instructions = list(model.wv.vocab)
# instructions = list(dmodel.wv.vocab)

# for i, instruction in enumerate(instructions):
#     pyplot.annotate(instruction, xy=(result[i, 0], result[i, 1]))

pyplot.show()










