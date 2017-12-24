from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
import gensim
import zipfile
import os
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

current_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = 'output'
data_directory2 = 'each'
filename = 'basicblock_by_file'
model_directory = 'model'
model_directory2 = 'bin_model'
modelname = filename+'_bin2vec.model'
modellistname = 'bin2vec_model_list'
# file_path = os.path.join(current_directory, data_directory, data_directory2)
file_path = os.path.join(current_directory, data_directory)
model_path = os.path.join(current_directory, model_directory, model_directory2, modelname)
model_list_path = os.path.join(current_directory, model_directory, modellistname)
each_zipfile = os.path.join(file_path, '_each_total.zip')


'''
Create Doc2Vec Model
'''


def read_data(filename):

    of = open(model_list_path, 'w')

    with zipfile.ZipFile(filename) as f:
        binList = [hash for hash in f.namelist()]
        for bin in binList:
            buf = str()
            for idx, line in enumerate(f.open(bin)):
                buf = buf + str(line.strip().decode('utf-8'))
            bfLabel = bin
            wl = bfLabel + ' ' + buf + '\n'
            of.write(wl)
            yield TaggedDocument(gensim.utils.simple_preprocess(buf.strip(), min_len=1, max_len=100), [bfLabel])
            print(bin, 'read, write completed')
    of.close()



print('each.zip file read & list make completed')

train_data = list(read_data(each_zipfile))
print('train_data get completed')

model = doc2vec.Doc2Vec(size=400, window=10, min_count=5, iter=1000, workers=4)
model.build_vocab(train_data)
model.save(model_path)

print('model make completed & saved')



'''
Load Doc2Vec Model
'''
#
# model = doc2vec.Doc2Vec.load(model_path)
# # docvec = model.docvecs['fff564d59deec80ad5fcc92867e07b69_17']
# docvec = model.docvecs['fff564d59deec80ad5fcc92867e07b69']
# # sims = model.docvecs.most_similar('02456317d10e2d769f704935d5b5b6cd_2137')
# sims = model.docvecs.most_similar('02456317d10e2d769f704935d5b5b6cd')
# # sims2 = model.most_similar('intthree')
# print(sims)
# # print(sims2)



