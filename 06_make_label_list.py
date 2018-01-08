from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.preprocessing import MinMaxScaler
import gensim
import os
import logging
import zipfile
import csv

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model_directory = 'model'
model_directory2 = 'bin_model'
filename = 'basicblock_by_file'
modelname = filename+'_bin2vec.model'
data_directory = 'output'
current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, model_directory, model_directory2, modelname)
output_filename = 'label_list.csv'
output_path = os.path.join(current_directory, model_directory, output_filename)
file_path = os.path.join(current_directory, data_directory)
mann_dataset_zip = os.path.join(file_path, 'mann_dataset.zip')
bin2vec_dataset_zip = os.path.join(file_path, 'bin2vec_dataset.zip')

'''
Create model/label_list.csv file
'''

# Create mann_dataset_list & bin2vec_dataset_list
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        fileList = [hash for hash in f.namelist()]

    return fileList


mann_dataset_list = read_data(mann_dataset_zip)
bin2vec_dataset_list = read_data(bin2vec_dataset_zip)


# # Create label_list [md5, detectname, mann/bin2vec/none]
# f = open('output/label.csv', 'r', encoding='utf-8')
# rdr = csv.reader(f)
#
# label_list = list()
# for line in rdr:
#     md5 = line[0].strip()
#     asd_name = line[3].strip()
#
#     # print(line)
#     if md5 in mann_dataset_list:
#         value_list = list()
#         value_list.append(md5)
#         value_list.append(asd_name)
#         value_list.append('mann')
#
#     elif md5 in bin2vec_dataset_list:
#         value_list = list()
#         value_list.append(md5)
#         value_list.append(asd_name)
#         value_list.append('bin2vec')
#
#     else:
#         value_list = list()
#         value_list.append(md5)
#         value_list.append(asd_name)
#         value_list.append('none')
#
#     label_list.append(value_list)
#
# f.close()
#
#
# # Create model/label_list.csv
# with open(output_path, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     for data in label_list:
#         writer.writerow(data)


'''
Load Doc2Vec Model
'''
model = doc2vec.Doc2Vec.load(model_path)

'''
Read mann_dataset.zip
Read bin2vec_dataset.zip
'''


def read_data(filename):

    sample_dict = dict()

    with zipfile.ZipFile(filename) as f:
        binList = [hash for hash in f.namelist()]

        for bin in binList:
            buf = str()
            bfLabel = bin

            for idx, line in enumerate(f.open(bin)):
                buf = buf + str(line.strip().decode('utf-8') + ' ')

            sample_dict[bfLabel] = buf.strip()

    return sample_dict




# train_data = list(read_data(each_zipfile))
# s_dict = read_data(mann_dataset_zip)
# print(len(s_dict.keys()))
# print(s_dict['ff4811ffc8c236f5955c0e0c1f3b49e4'].split())

# insts = s_dict['ef562c53e3d4e2d46549ae17497d6642'].split()

# infer_vec_list = list()
# size = len(s_dict.keys())
# infer_count = 1

# for key in s_dict.keys():
#     insts = s_dict[key].split()
#     infer_vec_list.append(model.infer_vector(insts, alpha=0.1, min_alpha=0.0001, steps=5))
#     print('infer complete', infer_count, '/', size, ':', key)
#     infer_count = infer_count + 1

# docvec = model.docvecs['fff564d59deec80ad5fcc92867e07b69_17']
# docvec = model.docvecs['ef562c53e3d4e2d46549ae17497d6642']
# docvec = model.infer_vector['ef562c53e3d4e2d46549ae17497d6642']
# print(docvec)
# sims = model.docvecs.most_similar('02456317d10e2d769f704935d5b5b6cd_2137')
# sims = model.docvecs.most_similar('02456317d10e2d769f704935d5b5b6cd')
# sims2 = model.most_similar('pushebp')
# print(sims)
# print(sims2)


'''
TSNE test
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.manifold import TSNE

# wv = model.wv.syn0
# vocabulary = model.wv.vocab
#
# tsne = TSNE(n_components=2, random_state=0)
# np.set_printoptions(suppress=True)
# Y = tsne.fit_transform(wv[:1000, :])
#
# plt.scatter(Y[:, 0], Y[:, 1])
# for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
#     plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
# plt.show()

label_file = 'E:/Works/temp/model/label_list_re.csv'

f = open(label_file, 'r', encoding='utf-8')
rdr = csv.reader(f)

file_list = list()

for line in rdr:
    md5 = line[0].strip()
    category = line[1]
    file_list.append(md5)

# vector value dictionary
value_dict = dict()
for f in file_list:
    value_dict[f] = model.docvecs[f]

# normalization
norm_value_dict = dict()
for key in value_dict.keys():
    x = value_dict[key]

    # v = x

    # v = x / np.linalg.norm(x)

    # max_value = np.max(x)    # normalization is important
    # if max_value > 0.:
    #     v = x / max_value

    numerator = x - np.min(x, 0)
    denominator = np.max(x, 0) - np.min(x, 0)
    v = numerator / (denominator + 1e-7)

    norm_value_dict[key] = v

# print(norm_value_dict)

# change list
c_norm_value_list = list()
for k in norm_value_dict.keys():
    x = norm_value_dict[k]
    c_norm_value_list.append(x)

t = np.array(c_norm_value_list)
print(t.shape)

tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
#
Y = tsne.fit_transform(t[:3200, :])
plt.scatter(Y[:, 0], Y[:, 1])
# for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
#     plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
plt.show()