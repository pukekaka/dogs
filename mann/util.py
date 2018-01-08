import numpy as np
import pandas as pd
from gensim.models import doc2vec
import logging
import zipfile
import csv
from sklearn.preprocessing import MinMaxScaler
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def one_hot_encode(x, dim):
    res = np.zeros(np.shape(x) + (dim, ), dtype=np.float32)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        res[it.multi_index][it[0]] = 1
        it.iternext()
    return res


def one_hot_decode(x):
    return np.argmax(x, axis=-1)


def read_data(zip_file):

    sample_dict = dict()

    with zipfile.ZipFile(zip_file) as f:
        binList = [hash for hash in f.namelist()]

        for bin in binList:
            buf = str()
            bfLabel = bin

            for idx, line in enumerate(f.open(bin)):
                buf = buf + str(line.strip().decode('utf-8') + ' ')

            sample_dict[bfLabel] = buf.strip()

    return sample_dict

class SampleDataLoader:
    def __init__(self, label_path, model_path, data_path, n_train_classes, n_test_classes):
        self.data = []
        self.filename_list = []
        self.label_path = label_path
        self.model_path = model_path
        self.n_train_classes = n_train_classes
        self.n_test_classes = n_test_classes
        self.processing_count = 0

        '''
        infer version
        '''
        # f = open(self.label_path, 'r', encoding='utf-8')
        # rdr = csv.reader(f)

        # Make mann list & dictionary
        # self.mann_dataset_list = list()
        # self.mann_dataset_dict = dict()
        # self.dataset_label_dict = dict()
        # for line in rdr:
        #     md5 = line[0]
        #     detect_name = line[1]
        #     dataset = line[2]
        #     if dataset == 'mann':
        #         self.mann_dataset_list.append(md5)
        #         if self.mann_dataset_dict.get(detect_name) is None:
        #             temp_list = list()
        #             temp_list.append(md5)
        #             self.mann_dataset_dict[detect_name] = temp_list
        #         else:
        #             self.mann_dataset_dict[detect_name].append(md5)

        # Move data <- mann_dataset
        # label_count = 0
        # for key in self.mann_dataset_dict.keys():
        #     self.data.append(self.mann_dataset_dict[key])
        #     self.dataset_label_dict[key] = label_count
        #     label_count = label_count + 1

        # Infer mann data file's vectors
        # self.model = doc2vec.Doc2Vec.load(self.model_path)
        # zipfile_path = 'E:/Project/PycharmProjects/dogs/output/mann_dataset.zip'
        # self.s_dict = read_data(zipfile_path)

        # self.infer_vec_dict = dict()
        # size = len(self.s_dict.keys())
        # infer_count = 1

        # for key in self.s_dict.keys():
        #     insts = self.s_dict[key].split()
        #     self.infer_vec_dict[key] = self.model.infer_vector(insts, alpha=0.1, min_alpha=0.0001, steps=5)
        #     print('infer complete', infer_count, '/', size, ':', key)
        #     infer_count = infer_count + 1

        '''
        infer version end
        '''

        '''
        test infer version
        '''
        #
        # f = open(self.label_path, 'r', encoding='utf-8')
        # rdr = csv.reader(f)
        #
        # # Make mann list & dictionary
        # self.mann_dataset_list = list()
        # self.mann_dataset_dict = dict()
        # self.dataset_label_dict = dict()
        # for line in rdr:
        #     md5 = line[0]
        #     detect_name = line[1]
        #     dataset = line[2]
        #     if dataset == 'mann':
        #         self.mann_dataset_list.append(md5)
        #         if self.mann_dataset_dict.get(detect_name) is None:
        #             temp_list = list()
        #             temp_list.append(md5)
        #             self.mann_dataset_dict[detect_name] = temp_list
        #         else:
        #             self.mann_dataset_dict[detect_name].append(md5)
        #
        # self.model = doc2vec.Doc2Vec.load(self.model_path)
        # zipfile_path = 'E:/Project/PycharmProjects/dogs/output/mann_dataset.zip'
        # self.s_dict = read_data(zipfile_path)
        #
        # category = 15
        # each_file = 20
        #
        # self.sample_mann_dataset_dict = dict()
        # self.sample_s_dict = dict()
        # sample_category_count = 0
        #
        # for md_key in self.mann_dataset_dict.keys():
        #     if sample_category_count < category:
        #         self.sample_mann_dataset_dict[md_key] = self.mann_dataset_dict[md_key]
        #
        #         # Move data <- mann_dataset
        #         hash_list = self.mann_dataset_dict[md_key]
        #         self.data.append(hash_list)
        #
        #         for hash in hash_list:
        #             self.sample_s_dict[hash] = self.s_dict[hash]
        #     sample_category_count = sample_category_count + 1

        # print(len(self.sample_s_dict.keys()))
        # size = category * each_file
        # infer_count = 1
        # self.infer_vec_dict = dict()

        # for key in self.sample_s_dict.keys():
        #     # insts = self.sample_s_dict[key].split()
        #     self.infer_vec_dict[key] = self.model.docvecs[key]
        #     # self.infer_vec_dict[key] = self.model.infer_vector(insts, alpha=0.1, min_alpha=0.0001, steps=5)
        #     print('infer complete', infer_count, '/', size, ':', key)
        #     infer_count = infer_count + 1

        '''
        test infer version end
        '''

        '''
        test version
        '''
        #
        # f = open(self.label_path, 'r', encoding='utf-8')
        # rdr = csv.reader(f)
        #
        # self.bin_dataset_list = list()
        # self.bin_dataset_dict = dict()
        # self.dataset_label_dict = dict()
        # label_count = 0
        #
        # for line in rdr:
        #     md5 = line[0]
        #     detect_name = line[1]
        #     dataset = line[2]
        #     if dataset == 'bin2vec':
        #         self.bin_dataset_list.append(md5)
        #         if self.bin_dataset_dict.get(detect_name) is None:
        #             temp_list = list()
        #             temp_list.append(md5)
        #             self.bin_dataset_dict[detect_name] = temp_list
        #         else:
        #             self.bin_dataset_dict[detect_name].append(md5)
        #
        # for key in self.bin_dataset_dict.keys():
        #     self.data.append(self.bin_dataset_dict[key])
        #     self.dataset_label_dict[key] = label_count
        #     label_count = label_count + 1
        #
        # category = 15
        # each_file = 20
        # get_count = 1
        # self.infer_vec_dict = dict()
        #
        # self.model = doc2vec.Doc2Vec.load(self.model_path)
        # for bin in self.bin_dataset_list:
        #     self.infer_vec_dict[bin] = self.model.docvecs[bin]
        #     print('get complete', get_count, '/', category * each_file, ':', bin)
        #     get_count = get_count + 1

        '''
        test version end
        '''

        '''
        new version
        '''

        self.model = doc2vec.Doc2Vec.load(self.model_path)

        for dirname, subdirname, filelist in os.walk(data_path):
            if filelist:
                self.data.append(
                    # [np.reshape(
                    #     np.array(Image.open(dirname + '/' + filename).resize(image_size), dtype=np.float32),
                    #     newshape=(image_size[0] * image_size[1])
                    #     )
                    #     for filename in filelist]
                    # [io.imread(dirname + '/' + filename).astype(np.float32) / 255 for filename in filelist]
                    [filename for filename in filelist]
                )

        # train_data=112 / test_data=48
        self.train_data = self.data[:n_train_classes]
        self.test_data = self.data[-n_test_classes:]

    def fetch_batch(self, n_classes, batch_size, seq_length, type='train', label_type='one_hot'):

        if type == 'train':
            data = self.train_data
        elif type == 'test':
            data = self.test_data

        # Random arrange in each 5
        # print(len(data))
        classes = [np.random.choice(range(len(data)), replace=False, size=n_classes) for _ in range(batch_size)]
        seq = np.random.randint(0, n_classes, [batch_size, seq_length])

        seq_bin = [[self.getVec(data[classes[i][j]][np.random.randint(0, len(data[classes[i][j]]))])
                    for j in seq[i, :]]
                    for i in range(batch_size)]

        if label_type == 'one_hot':
            seq_encoded = one_hot_encode(seq, n_classes)
            seq_encoded_shifted = np.concatenate(
                [np.zeros(shape=[batch_size, 1, n_classes]), seq_encoded[:, :-1, :]], axis=1
            )

        return seq_bin, seq_encoded_shifted, seq_encoded

    def getVec(self, label):
        # x = np.array(self.infer_vec_dict[label])
        x = np.array(self.model.docvecs[label])
        '''
        pandas Nomalization
        '''
        # x = self.infer_vec_dict[label]
        #
        # # normalization
        # p_data = pd.DataFrame(data=x)
        # pd_norm = (p_data - p_data.mean()) / (p_data.max() - p_data.min())
        #
        # v = np.array(pd_norm[0])

        '''
        sample Normalization
        '''
        # max_value = np.max(x)    # normalization is important
        # if max_value > 0.:
        #     v = x / max_value

        '''
        numpy Normalization
        '''
        # v = x / np.linalg.norm(x)

        '''
        Scaler
        '''
        numerator = x - np.min(x, 0)
        denominator = np.max(x, 0) - np.min(x, 0)
        v = numerator / (denominator + 1e-7)
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # v = scaler.fit_transform([x])
        # print(v)
        # print(np.array(v).shape)
        # print (v)

        # v = x
        # print(v)
        return v


# from mann import param
#
# iv = param.init_value()
# data_loader = SampleDataLoader(
#     label_path=iv.label_path,
#     model_path=iv.model_path,
#     data_path=iv.data_path,
#     n_train_classes=iv.n_train_classes,
#     n_test_classes=iv.n_test_classes
# )
# x_inst, x_label, y = data_loader.fetch_batch(iv.n_classes, iv.batch_size, iv.seq_length, type='train')