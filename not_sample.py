import os
import csv
import shutil

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

print("%0.7f" % 5e-4)