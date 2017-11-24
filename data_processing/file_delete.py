import os


path = 'E:/Works/Data/samples/malwares/Cerber'


filename_dict = {}

for root, dirs, files in os.walk(path):
    for fn in files:

        try:
            n = os.path.getsize(root+'/'+fn)
            size = n / 1024
            filename_dict[fn] = size

        except os.error:
            print('error')

inv_map = {v: k for k, v in filename_dict.items()}
file_list = {v: k for k, v in inv_map.items()}

count = 0
for root, dirs, files in os.walk(path):
    for fn in files:
        if not file_list.get(fn):
            count += 1
            # print (root+'/'+fn)
            os.remove(root+'/'+fn)

print(count)
# for key in filename_dict.keys():


# print(len(inv_map))

#
