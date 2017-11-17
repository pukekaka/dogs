import json
from collections import OrderedDict
#
# data = {}
# event = {}
#
# event['type'] = 1
# event['value'] = 3
#
# for i in range(1, 10):
#     data[i] = event
#
# print json.dumps(data, sort_keys=True, indent=4)

import os

data_directory = 'C:/Data/test_output/'

asmf_extension = 'asmf'
bblist_extension = 'bblist'
filelist = []
asmf_filelist = []
bblist_filelist = []

filecount = 0
asmf_count = 0
bblist_count = 0

for root, dirs, files in os.walk(data_directory):
    for file in files:

        filelist.append(file)
        filecount += 1

        filename = file.split('.')
        extension_loc = len(filename) - 1
        if asmf_extension == filename[extension_loc] :
            asmf_filelist.append(file)
            asmf_count += 1

        if bblist_extension == filename[extension_loc] :
            bblist_filelist.append(file)
            bblist_count += 1

print ('file count :', filecount)
print ('asmf count :', asmf_count)
print ('bblist count :', bblist_count)

print ('filelist count', len(filelist))
print ('asmf filelist count', len(asmf_filelist))
print ('bblist filelist count', len(bblist_filelist))

print(asmf_filelist)

binary = []
bbl = []
bbl_list = []
func = []
func_list = []






# group_data = OrderedDict()
# albums = OrderedDict()
#
# group_data["binary"] = 'xxxx'
# group_data["flist"] = ['sowon', "yerin", "eunha", "youju"]
#
# print(json.dumps(group_data, ensure_ascii=False, indent=4))