import zipfile
import os
import pandas as pd

current_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = 'temp'
file_path = os.path.join(current_directory, data_directory)
label_list_zipfile = os.path.join(file_path, 'label_list.zip')
each_zipfile = os.path.join(file_path, 'each.zip')



'''
Create Label_list - label_list
temp/label_list.zip
'''


def read_label_file(label_zip_path):
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
                    file_info.append(data[3].strip()) # TS
                    file_info.append(data[4].strip()) # ASD
                    file_info.append(file.split('.')[0].strip()) # category
                    file_dict[data[1].strip()] = file_info
                    count = count + 1
            print(file, count)
    return file_dict


label_dict = read_label_file(label_list_zipfile)
print(len(label_dict))


'''
Create each.zip, label_list - el_list
temp/each.zip 
'''


def read_each_file(each_file_path, label_dict):
    each_list = list()
    with zipfile.ZipFile(each_file_path) as f:
        each_file_List = [c for c in f.namelist()]
        for file in each_file_List:
            file_info = list()
            try:
                size = f.getinfo(file).file_size # file size 0 except
                if size != 0:
                    category = label_dict[file][2] # get category
                    file_info.append(file)
                    file_info.append(category)
                    each_list.append(file_info)
            except:
                print('Not Exist', file)
    return each_list


each_list = read_each_file(each_zipfile, label_dict)
print(each_list)


label = list()
for each in each_list:
    label.append(each[1])
li = list(set(label))
print(len(li))
