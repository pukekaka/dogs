import numpy as np
import os
from gensim.models import doc2vec
import logging
import zipfile

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

current_directory = os.path.dirname(os.path.abspath(__file__))
filename = 'basicblock_by_file'
model_directory = 'model'
model_directory2 = 'bin_model'
modelname = filename + '_bin2vec.model'
modellistname = 'bin2vec_model_list'
model_path = os.path.join(current_directory, model_directory, model_directory2, modelname)
model_list_path = os.path.join(current_directory, model_directory, modellistname)

def one_hot_encode(x, dim):
    res = np.zeros(np.shape(x) + (dim, ), dtype=np.float32)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        res[it.multi_index][it[0]] = 1
        it.iternext()
    return res

def one_hot_decode(x):
    return np.argmax(x, axis=-1)


class ModelData:
    def __init__(self):
        self.current_directory = os.path.dirname(os.path.abspath(__file__))
        self.data_directory = 'temp'
        self.file_path = os.path.join(self.current_directory, self.data_directory)
        # self.label_list_zipfile = os.path.join(self.file_path, 'label_list.zip')
        self.label_list_zipfile = 'E:/Project/PycharmProjects/dogs/temp/label_list.zip'
        # zip_path = 'E:/Project/PycharmProjects/dogs/temp/label_list.zip'
        # self.each_zipfile = os.path.join(self.file_path, 'each.zip')
        self.each_zipfile = 'E:/Project/PycharmProjects/dogs/temp/each.zip'
        self.filename = 'basicblock_by_file'
        self.model_directory = 'model'
        self.model_directory2 = 'bin_model'
        self.model_name = self.filename + '_bin2vec.model'
        # self.model_path = os.path.join(self.current_directory, self.model_directory, self.model_directory2, self.model_name)
        self.model_path = 'E:/Project/PycharmProjects/dogs/model/bin_model/basicblock_by_file_bin2vec.model'

    def read_label_file(self, label_zip_path):
        file_dict = dict()
        with zipfile.ZipFile(label_zip_path) as f:
            category_file_List = [c for c in f.namelist()]
            # category_List = [c.split('.')[0] for c in f.namelist()]
            for file in category_file_List:
                count = 0
                for idx, line in enumerate(f.open(file)):
                    file_info = list()
                    if idx != 0:
                        data = line.decode('utf-8').split('\t')
                        # file_info.append(data[1].strip()) # hash
                        file_info.append(data[3].strip())  # TS
                        file_info.append(data[4].strip())  # ASD
                        file_info.append(file.split('.')[0].strip())  # category
                        file_dict[data[1].strip()] = file_info
                        count = count + 1
                # print(file, count)
        return file_dict

    def make_class_dict(self, hl_dict):
        category_hash_dict = dict()
        category_list = list(set([hl_dict[hl][2] for hl in hl_dict.keys()]))
        for category in category_list:
            hash_vlist = list()
            for hl in hl_dict.keys():
                cv = hl_dict[hl][2]
                if cv == category:
                    hash_vlist.append(hl)
            category_hash_dict[category] = hash_vlist
        return category_hash_dict

    def get_data(self):
        self.label_dict = self.read_label_file(self.label_list_zipfile)
        ch_dict = self.make_class_dict(self.label_dict)

        return ch_dict

class SampleDataLoader:
    def __init__(self, zip_dir, model_dir, n_train_classes, n_test_classes):
        self.data = []
        self.filename_list = []
        self.md = ModelData()
        self.sample_data_dict = self.md.get_data()
        self.model_path = 'E:/Project/PycharmProjects/dogs/model/bin_model/basicblock_by_file_bin2vec.model'
        self.model = doc2vec.Doc2Vec.load(self.model_path)

        for category in self.sample_data_dict.keys():
            temp_list = list()
            for h in self.sample_data_dict[category]:
                try:
                    if len(temp_list) < 20:
                        docvec = self.model.docvecs[h]
                        temp_list.append(h)
                        # print(temp_list)
                except:
                    pass
            self.data.append(temp_list)
        # zip_file = zip_dir + 'each.zip'
        # with zipfile.ZipFile(zip_file) as f:
        #     self.filename_list = [h_file for h_file in f.namelist()]
        #
        # for self.filename in self.filename_list:
        #     self.data.append(self.model.docvecs[self.filename])
        #
        #
        # print(np.array(self.data).shape)
        # print(len(self.data[0]))
        self.ndata = []
        for d in self.data:
            if len(d) == 20:
                self.ndata.append(d)
        self.data = self.ndata

        self.train_data = self.data[:n_train_classes]
        self.test_data = self.data[-n_test_classes:]

        # print(np.array(self.data).shape)
        # print(len(self.train_data))
        # print(len(self.test_data))


    def fetch_batch(self, n_classes, batch_size, seq_length, type='train', label_type='one_hot'):

        if type == 'train':
            data = self.train_data
        elif type == 'test':
            data = self.test_data


        classes = [np.random.choice(range(len(data)), replace=False, size=n_classes) for _ in range(batch_size)]
        # print(np.array(classes).shape)
        # print(classes)

        seq = np.random.randint(0, n_classes, [batch_size, seq_length])
        # print(np.array(seq).shape)
        # print(seq)
        seq_bin = [[self.docVec(data[classes[i][j]][np.random.randint(0, len(data[classes[i][j]]))])
                    for j in seq[i, :]]
                    for i in range(batch_size)]
        # print(len(seq_bin[0][0]))

        if label_type == 'one_hot':
            seq_encoded = one_hot_encode(seq, n_classes)
            seq_encoded_shifted = np.concatenate(
                [np.zeros(shape=[batch_size, 1, n_classes]), seq_encoded[:, :-1, :]], axis=1
            )

        return seq_bin, seq_encoded_shifted, seq_encoded

    def docVec(self, h):

        d2v = np.array(self.model.docvecs[h])
        max_value = np.max(d2v)  # normalization
        if max_value > 0.:
            d2v = d2v / max_value

        return d2v

zip_path = 'E:/Project/PycharmProjects/dogs/temp/'
model_path = 'E:/Project/PycharmProjects/dogs/model/bin_model/basicblock_by_file_bin2vec.model'

data_loader = SampleDataLoader(
            zip_dir=zip_path,
            model_dir=model_path,
            n_train_classes=200,
            n_test_classes=61
        )
#
x_inst, x_label, y = data_loader.fetch_batch(5, 128, 50, type='train')
# data_loader.fetch_batch(5, 128, 50, type='train')



