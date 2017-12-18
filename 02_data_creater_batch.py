import shutil
import os


'''
Data folder -> if(json file > 50MB) -> Delete the hash 
'''
current_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = 'data2'
files_path = os.path.join(current_directory, data_directory)

bb_extension = 'json'
bblist_extension = 'bblist'

deletefile_list = []

try:
    for root, dirs, files in os.walk(files_path):
        for file in files:

            filename_split = file.split('.')
            extension_loc = len(filename_split) - 1
            if bb_extension == filename_split[extension_loc]:
                n = os.path.getsize(root+'/'+file) / (1024.0 * 1024.0)
                if n > 50:
                    deletefile = root+'/'+filename_split[0]+'.'+bb_extension
                    deletefile2 = root + '/' + filename_split[0]+ '.'+ bblist_extension
                    deletefile_list.append(deletefile)
                    deletefile_list.append(deletefile2)

except os.error:
    print("파일이 없거나 에러입니다.")



for deletefile in deletefile_list:
    print('delete file: ', deletefile)
    os.remove(deletefile)



'''
Data folder -> 0,1,2,3...folder -> each 100 files...
'''

current_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = 'data2'
files_path = os.path.join(current_directory, data_directory)

file_count = 0
file_processing_counting = 50
folder_list = list()

for root, dirs, files in os.walk(files_path):
    for filename in files:
        file_count = file_count + 1

file_count = int(file_count / 2)

folder_count = int(file_count / file_processing_counting) + 1
for folder in range(folder_count):
    dirname = files_path + '/' + str(folder)
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
        folder_list.append(folder)


folder_name = 0
moving_file_count = 0

for root, dirs, files in os.walk(files_path):
    for filename in files:
        if moving_file_count < file_processing_counting*2:
            dirname = files_path + '/' + str(folder_name)
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            shutil.move(files_path + '/' + filename, dirname)
            moving_file_count = moving_file_count + 1

        else:
            moving_file_count = 0
            folder_name = folder_name + 1
            dirname = files_path + '/' + str(folder_name)
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            shutil.move(files_path + '/' + filename, dirname)
            moving_file_count = moving_file_count + 1
