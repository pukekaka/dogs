from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
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


# Create label_list [md5, detectname, mann/bin2vec/none]
f = open('output/label.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)

label_list = list()
for line in rdr:
    md5 = line[0].strip()
    asd_name = line[3].strip()

    # print(line)
    if md5 in mann_dataset_list:
        value_list = list()
        value_list.append(md5)
        value_list.append(asd_name)
        value_list.append('mann')

    elif md5 in bin2vec_dataset_list:
        value_list = list()
        value_list.append(md5)
        value_list.append(asd_name)
        value_list.append('bin2vec')

    else:
        value_list = list()
        value_list.append(md5)
        value_list.append(asd_name)
        value_list.append('none')

    label_list.append(value_list)

f.close()


# Create model/label_list.csv
with open(output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for data in label_list:
        writer.writerow(data)



'''
Load Doc2Vec Model
'''
# model = doc2vec.Doc2Vec.load(model_path)

















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
# s_dict = read_data(each_zipfile)
# print(len(s_dict.keys()))
# print(s_dict['ff4811ffc8c236f5955c0e0c1f3b49e4'].split())

# buf_list = s_dict['ef562c53e3d4e2d46549ae17497d6642'].split()

# min_len=1, max_len=100
# test = model.infer_vector(buf_list, alpha=0.1, min_alpha=0.0001, steps=5)
# print(test)

# docvec = model.docvecs['fff564d59deec80ad5fcc92867e07b69_17']
# docvec = model.docvecs['ef562c53e3d4e2d46549ae17497d6642']
# docvec = model.infer_vector['ef562c53e3d4e2d46549ae17497d6642']
# print(docvec)
# sims = model.docvecs.most_similar('02456317d10e2d769f704935d5b5b6cd_2137')
# sims = model.docvecs.most_similar('02456317d10e2d769f704935d5b5b6cd')
# sims2 = model.most_similar('pushebp')
# print(sims)
# print(sims2)

