import json
import os

data_directory = 'E:/Works/Data/samples/output/'
output_directory = 'E:/Works/Data/samples/output_c/'
output_filename = 'opcode_opnd'

bb_extension = 'json'
bblist_extension = 'bblist'
bb_filenamelist = []
bblist_filenamelist = []

print('=> filename read start')

for root, dirs, files in os.walk(data_directory):
    for file in files:

        filename_split = file.split('.')
        extension_loc = len(filename_split) - 1
        if bb_extension == filename_split[extension_loc]:
            bb_filenamelist.append(root+file)

        elif bblist_extension == filename_split[extension_loc]:
            bblist_filenamelist.append(root+file)

print('=> filename read end')

bb_data_list = []
bblist_data_list = []
count = 1

print('=> json object read start')

for bb_filename in bb_filenamelist:
    bb_data_file = open(bb_filename, encoding="utf-8")
    bb_data = json.load(bb_data_file)
    bb_data_list.append(bb_data)
    print(count, 'complete', bb_filename)
    count += 1

print('=> json object read end')
print('=> opcode + opnd make start')

word_list = []
bbcount = 1
for bb_data in bb_data_list:
    for inst in bb_data['insts']:
        opcode = inst['opcode']
        opnd1 = inst['opnd1']
        opnd2 = inst['opnd2']
        opnd3 = inst['opnd3']
        word = opcode.strip()+opnd1.strip()+opnd2.strip()+opnd3.strip()
        word = word.strip()
        word_list.append(word)
    print(bbcount, 'complete', bb_data['filename'])
    bbcount += 1

print('=> opcode + opnd make end')
print('=> opcode + opnd write start')

output_file = open(output_directory + output_filename, 'w')
output_file.write(" ".join(word_list))
output_file.close()

print('=> opcode + opnd write end')