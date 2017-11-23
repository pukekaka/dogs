#-*-coding:utf-8-*-
import json
import os

data_directory = 'E:/Works/Data/samples/output/'
output_directory = 'E:/Works/Data/samples/output_c/'
output_filename = 'basicblock_by_line'

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
print('-basicblock')
for bb_filename in bb_filenamelist:
    bb_data_file = open(bb_filename, encoding="utf-8")
    bb_data = json.load(bb_data_file)
    bb_data_list.append(bb_data)
    print('basicblock', count, 'complete', bb_filename)
    count += 1

print('-basicblocklist')
count = 1
for bblist_filename in bblist_filenamelist:
    try:
        bblist_data_file = open(bblist_filename, encoding="utf-8-sig")
        bblist_data = json.load(bblist_data_file)
        bblist_data_list.append(bblist_data)
        print('basicblocklist', count, 'complete', bblist_filename)
        count += 1
    except:
        print('basicblocklist', count, 'error', bblist_filename)



print('=> json object read end')

print('=> start data dict make')
bb_data_list_dict ={}
for bb_data in bb_data_list:
    filename = bb_data['filename']
    bb_data_list_dict[filename] = bb_data

# print(bb_data_list_dict)

print('=> start line list make')

# example : {'filename':'line list'}
def search_line(filename, startaddress):
    for bb_data in bb_data_list:
        if filename == bb_data['filename']:
            for inst in bb_data['insts']:
                addr = inst['addr']
                line = inst['line']
                if startaddress == addr:
                    return line

bblist_line_dict={}

count = 1
for bblist_data in bblist_data_list:
    bb_start_line_list = []
    filename = bblist_data['filename']
    for bblist in bblist_data['bblist']:
        startaddress = bblist['start']
        line = search_line(filename, startaddress)
        if line != -1:
            bb_start_line_list.append(line)

    bblist_line_dict[filename] = bb_start_line_list
    print('line list', count, 'complete', filename)
    count += 1

print('=> end line list make')


print('=> start end range list make')
count = 1
# example : {'filename' : 'range list'}
bb_range_list = {}
for bblist_line_key in bblist_line_dict.keys():
    prev = 0
    linelist = bblist_line_dict[bblist_line_key]
    range_list = []
    size = len(linelist)
    for i, bblist_data_line in enumerate(linelist):
        if i == 0:
            prev = bblist_data_line
        else:
            data = str(prev)+':' + str(bblist_data_line)
            prev = bblist_data_line
            range_list.append(data)
    bb_range_list[bblist_line_key] = range_list
    print('range list', count, 'complete', bblist_line_key)
    count+=1

print('=> start end range list make end')

print('=> start result')

def instlistbyline(insts, inst_range):
    word_list = []

    iline = inst_range.split(':')
    if len(iline) > 1:
        startiline = int(iline[0])
        endiline = int(iline[1])

    for inst in insts:
        line = inst['line']
        if line >= startiline and line < endiline:
            opcode = inst['opcode']
            opnd1 = inst['opnd1']
            opnd2 = inst['opnd2']
            opnd3 = inst['opnd3']
            word = opcode.replace(" ", "") + opnd1.replace(" ", "") + opnd2.replace(" ", "") + opnd3.replace(" ", "")
            # word = opcode.replace(" ", "") + opnd1.strip() + opnd2.strip() + opnd3.strip()
            # sentence.replace(" ", "")
            word = word.strip()
            word_list.append(word)
    return word_list

line_list =[]
count = 1
for bb_range_key in bb_range_list.keys():
    for bb_data in bb_data_list:
        if bb_data['filename'] == bb_range_key:
            for bb_range in bb_range_list[bb_range_key]:
                try:
                    line_list.append(instlistbyline(bb_data['insts'], bb_range))
                except:
                    print('error', bb_data['filename'])
    print('result', count, 'complete', bb_range_key)
    count += 1


print('=> end result')
print('=> file write start')

output_file = open(output_directory + output_filename, 'w')
resultlist = []
for line in line_list:
    oneline = " ".join(line)
    resultlist.append(oneline)

output_file.write("\n".join(resultlist))
output_file.close()

print('=> file write end')