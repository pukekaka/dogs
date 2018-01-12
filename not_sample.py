import os
import csv
import shutil
import array

# label_file = 'E:/Works/temp/model/label_list_re.csv'
# file_folder = 'E:/Works/temp/model/mann_dataset_folder'
#
# f = open(label_file, 'r', encoding='utf-8')
# rdr = csv.reader(f)
#
# category_set = set()
# category_dict = dict()
#
# for line in rdr:
#     category = line[1]
#     category_set.add(category)
#
# for category in category_set:
#     path = os.path.join(file_folder, category)
#     if not os.path.exists(path):
#         os.makedirs(path)
#
# f2 = open(label_file, 'r', encoding='utf-8')
# rdr2 = csv.reader(f2)
#
# for line in rdr2:
#     md5 = line[0]
#     category_folder = line[1]
#     src = os.path.join(file_folder, md5)
#     dest = os.path.join(file_folder, category_folder)
#     shutil.move(src, dest)
#     print('ok')

from gensim.models import doc2vec
import numpy as np
from PIL import Image
import numpy as np
import zipfile
from scipy import misc

path = 'E:/Project/PycharmProjects/dogs/model/bin_model/basicblock_by_file_bin2vec.model'
label_file = 'E:/Works/temp/model/label_list_re.csv'

f = open(label_file, 'r', encoding='utf-8')
rdr = csv.reader(f)

hash_list = list()
model = doc2vec.Doc2Vec.load(path)

'''
doc2vec -> Image
'''
#
# count = 1
# for line in rdr:
#     hash = line[0]
#     x = np.array(model.docvecs[hash])
#     w, h = 20, 20
#     data = np.zeros((h, w), dtype=np.uint8)
#     data = np.reshape(x, (-1, 20))
#     img = Image.fromarray(data, 'L')
#     filename = hash+'.png'
#     img.save('E:/Works/png/'+filename)
#     print('complete', count,'/3200')
#     count = count + 1


'''
Binary -> Image
'''

# data_folder = 'E:/Works/png2/md/'
# save_folder = 'E:/Works/png2/'
# # mann_dataset_filename = 'mann_dataset.zip'
#
# count = 1
# for (path, dir, files) in os.walk(data_folder):
#     for filename in files:
#         path = data_folder+filename
#         f = open(path, 'rb')
#         ln = os.path.getsize(path)
#
#         width = int(ln**0.5)
#         # width = 256
#
#         rem = ln % width
#
#         a = array.array("B")
#
#         a.fromfile(f, ln-rem)
#         # print(len(a)/width)
#         f.close()
#
#         g = np.reshape(a, (int(len(a)/width), width))
#         g = np.uint8(g)
#         sp = save_folder+filename+'.png'
#         # print(path)
#         misc.imsave(sp, g)
#         print('complete', count, '/3200')
#         count = count + 1


'''
temp code
'''
ord_dataset = 'E:/Works/temp/model/m_dataset'
png_dataset = 'E:/Works/temp/model/m_dataset_p'

for line in rdr:
    hash = line[0]
    category = line[1]
    directory = os.path.join(png_dataset, category)

    if not os.path.exists(directory):
        os.makedirs(directory)

    src = os.path.join(png_dataset, hash+'.png')
    dest = os.path.join(png_dataset, directory, hash)
    shutil.move(src, dest)

'''
change gray -> binary image
'''

# import cv2
#
# data_folder = 'E:/Works/png/'
# save_folder = 'E:/Works/png3/'
#
# count = 1
# for (path, dir, files) in os.walk(data_folder):
#     for filename in files:
#         path = data_folder+filename
#         im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#         thresh = 127
#         im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
#         sp = save_folder + filename
#         cv2.imwrite(sp, im_bw)
#         print('complete', count, '/3200')
#         count = count + 1