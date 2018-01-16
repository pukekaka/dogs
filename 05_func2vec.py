from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
import gensim
import os
import logging

'''
test
'''
data_folder = 'E:/Works/Data2/apt_lastoutput'


word_set = set()
inst_set = set()
inst_list = list()

count = 1
for (path, dir, files) in os.walk(data_folder):
    for file in files:
        p = os.path.join(data_folder, file)
        f = open(p)
        lines = f.readlines()
        for line in lines:
            inst_set.add(line.strip())
            inst_list.append(line.strip())
            words = line.split()
            for word in words:
                word_set.add(word)
        print(count, 'complete', file)
        count = count + 1

print('word_set', len(word_set))
print('inst_set', len(inst_set))
print('inst_list', len(inst_list))



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

current_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = 'E:/Works/Data2/apt_lastoutput_zip'
filename = 'basicblock_by_func'
model_directory = 'model'
model_directory2 = 'test_model'
modelname = filename+'_test.model'
modellistname = 'test_model_list'
file_path = os.path.join(current_directory, data_directory)
model_path = os.path.join(current_directory, model_directory, model_directory2, modelname)
model_list_path = os.path.join(current_directory, model_directory, modellistname)
each_zipfile = os.path.join(file_path, 'apt_lastoutput.zip')

st = 'leagregmrbid leagregmrbid pushgreg pushgreg pushiv pushgreg calldmr'
st2 = 'addgregiv leagregmrbid leagregmrbid leagregmrbid pushgreg pushgreg pushiv pushgreg calldmr'

def read_data2(filename):

    of = open(model_list_path, 'w')
    size = len(inst_set)
    label = 0
    for inst in inst_set:
        wl = str(label) + ' ' + inst +'\n'
        of.write(wl)
        # il = inst.strip()
        # for i in il:
        yield TaggedDocument(gensim.utils.simple_preprocess(inst, min_len=3, max_len=100), [str(label)])
        # print('read, write completed', label, '/', size)
        label = label + 1

    of.close()


num = str()
num2 = str()

label = 0
for inst in inst_set:
    if inst.strip() == st:
        num = label
    if inst.strip() == st2:
        num2 = label
    label = label + 1


train_data = list(read_data2(each_zipfile))
print('train_data get completed')

model = doc2vec.Doc2Vec(size=20, window=3, min_count=5, iter=1000, workers=4)
model.build_vocab(train_data)

sims = model.most_similar('pushgreg')
sims2 = model.docvecs.most_similar(str(num))
print('sim', sims)
print('sim2', sims2)
print('num2', num2)

# model.save(model_path)

print('model make completed & saved')


'''
Load Doc2Vec Model
'''

# leagregmrbid leagregmrbid pushgreg pushgreg pushiv pushgreg calldmr
# movgregmrbid movmrbidgreg movmrbigreg movgregmrbid movmrbidgreg movmrbidgreg
test_inst = ['leagregmrbid', 'leagregmrbid', 'pushgreg', 'pushgreg', 'pushiv', 'pushgreg', 'calldmr']
# popedi popesi oreaxeax popebx retn
# test_inst = ['popedi', 'popesi', 'oreaxeax', 'popebx', 'retn']

# model = doc2vec.Doc2Vec.load(model_path)
# sims = model.infer_vector(test_inst, alpha=0.1, min_alpha=0.0001, steps=5)
# # docvec = model.docvecs['fff564d59deec80ad5fcc92867e07b69_17']
# docvec = model.docvecs['fff564d59deec80ad5fcc92867e07b69']
# sims = model.most_similar('pushgreg')
# sims = model.most_similar(test_inst)
# sims = model.docvecs.most_similar(str(3300))
# sims2 = model.docvecs[str(35335)]
# # sims2 = model.most_similar('intthree')

# print(sims)
# print(sims2)

import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy import spatial

# numerator = sims - np.min(sims, 0)
# denominator = np.max(sims, 0) - np.min(sims, 0)
# re_sims = numerator / (denominator + 1e-7)
#
# numerator = sims2 - np.min(sims2, 0)
# denominator = np.max(sims2, 0) - np.min(sims2, 0)
# re_sims2 = numerator / (denominator + 1e-7)


# sim3 = 1 - spatial.distance.cosine(sims, sims2)
# print(sim3)



# cos_sim = dot(sims, sims2)/(norm(sims)*norm(sims2))
#
# print(cos_sim)


# # print(sims2)