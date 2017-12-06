import numpy as np
import os
from gensim.models import doc2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

current_directory = os.path.dirname(os.path.abspath(__file__))
filename = 'basicblock_by_file'
model_directory = 'model'
model_directory2 = 'bin_model'
modelname = filename + '_bin2vec.model'
modellistname = 'bin2vec_model_list'
model_path = os.path.join(current_directory, model_directory, model_directory2, modelname)
model_list_path = os.path.join(current_directory, model_directory, modellistname)

def generate_random_strings(batch_size, seq_length, vector_dim):
    return np.random.randint(0, 2, size=[batch_size, seq_length, vector_dim]).astype(np.float32)


def one_hot_encode(x, dim):
    res = np.zeros(np.shape(x) + (dim, ), dtype=np.float32)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        res[it.multi_index][it[0]] = 1
        it.iternext()
    return res


def one_hot_decode(x):
    return np.argmax(x, axis=-1)


class MalwareDataLoader:

    def __init__(self, data_dir, image_size=(20, 20), n_train_classses=1200, n_test_classes=423):


    def load_model(self):
        model = doc2vec.Doc2Vec.load(model_path)
        # docvec = model.docvecs['fff564d59deec80ad5fcc92867e07b69_17']
        docvec = model.docvecs['fff564d59deec80ad5fcc92867e07b69']
        # sims = model.docvecs.most_similar('02456317d10e2d769f704935d5b5b6cd_2137')
        sims = model.docvecs.most_similar('02456317d10e2d769f704935d5b5b6cd')
        # sims2 = model.most_similar('intthree')
        print(sims)
        # print(sims2)