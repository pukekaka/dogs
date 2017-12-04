from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
import os
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

current_directory = os.path.dirname(os.path.abspath(__file__))
filename = 'basicblock_by_file'
model_directory = 'model'
model_directory2 = 'func_model'
modelname = filename+'_func2vec.model'
modellistname = 'func2vec_model_list'
model_path = os.path.join(current_directory, model_directory, model_directory2, modelname)
model_list_path = os.path.join(current_directory, model_directory, modellistname)

model = doc2vec.Doc2Vec.load(model_path)
funcvec = model.docvecs['fff564d59deec80ad5fcc92867e07b69_17']
# sims = model.docvecs.most_similar('02456317d10e2d769f704935d5b5b6cd_2137')
# sims2 = model.most_similar('intthree')
print(funcvec)
# print(sims2)
