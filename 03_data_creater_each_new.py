#-*-coding:utf-8-*-
import json
import os
import shutil
import inflect


suffix_filename = '_bb_by_line'
all_jb_fp = 'E:/Works/Data2/all_jsonbblist'
apt_jb_fp = 'E:/Works/Data2/apt_jsonbblist'
sample_fp = 'E:/Works/Data2/apt_sample'

# sample_fl = list()
# for (path, dir, files) in os.walk(sample_fp):
#     for filename in files:
#         sample_fl.append(filename)
#
#
# count = 1
# for dir_seq in os.listdir(all_jb_fp):
#     temp_folder = os.path.join(all_jb_fp, dir_seq)
#
#     for (path, dir, files) in os.walk(temp_folder):
#         for file in files:
#             filename_split = file.split('.')
#             filename = filename_split[0]
#             if filename in sample_fl:
#                 src = os.path.join(temp_folder, file)
#                 dest = os.path.join(apt_jb_fp, file)
#                 shutil.move(src, dest)
#                 print(count, '이동', file)
#                 count = count + 1

bb_extension = 'json'
bblist_extension = 'bblist'
bb_filenamelist = []
bblist_filenamelist = []

for root, dirs, files in os.walk(apt_jb_fp):
    for file in files:

        filename_split = file.split('.')
        extension_loc = len(filename_split) - 1
        if bb_extension == filename_split[extension_loc]:
            bb_filenamelist.append(root+'/'+file)

        elif bblist_extension == filename_split[extension_loc]:
            bblist_filenamelist.append(root+'/'+file)

print('=> filename read end')


bb_data_list = []
bblist_data_list = []
count = 1

print('=> json object read start')
print('-basicblock')
for bb_filename in bb_filenamelist:
    try:
        bb_data_file = open(bb_filename, encoding="utf-8-sig")
        bb_data = json.load(bb_data_file)
        bb_data_list.append(bb_data)
        print('basicblock', count, 'complete', bb_filename)
    except:
        print('basicblock', count, 'error', bb_filename)
    count += 1

print('-basicblocklist')

count = 1
for bblist_filename in bblist_filenamelist:
    try:
        bblist_data_file = open(bblist_filename, encoding="utf-8-sig")
        bblist_data = json.load(bblist_data_file)
        bblist_data_list.append(bblist_data)
        print('basicblocklist', count, 'complete', bblist_filename)
    except:
        print('basicblocklist', count, 'error', bblist_filename)
    count += 1

print('=> json object read end')


bb_data_list_dict ={}
for bb_data in bb_data_list:
    filename = bb_data['filename']
    bb_data_list_dict[filename] = bb_data

print('=> bb_block data dict make end')


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


opndtype_list = ['Nop', 'Reg', 'Mem', 'Phrase', 'Displ', 'Imm', 'Far', 'Near', 'Nop2']
p = inflect.engine()


def changeString(string):
    stringlist = list(str(string))
    changestringlist = []
    for item in stringlist:
        if item.isdigit():
            word = p.number_to_words(item, group=1)
            changestringlist.append(word)
        else:
            changestringlist.append(item)
    result = "".join(changestringlist)
    remove_specific_word_result = result.translate({ord(c): "" for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"})
    # print(result)
    return remove_specific_word_result

def changeType(optype):
    if optype == 1:
        return 'greg'
    elif optype == 2:
        return 'dmr'
    elif optype == 3:
        return 'mrbi'
    elif optype == 4:
        return 'mrbid'
    elif optype == 5:
        return 'iv'
    elif optype == 6:
        return 'ifa'
    elif optype == 7:
        return 'ina'
    else:
        return ''

# o_void  =      0  # No Operand -> NOP
# o_reg  =       1  # General Register (al,ax,es,ds...) -> GR
# o_mem  =       2  # Direct Memory Reference  (DATA) -> DMR
# o_phrase  =    3  # Memory Ref [Base Reg + Index Reg] -> MRBI
# o_displ  =     4  # Memory Reg [Base Reg + Index Reg + Displacement] -> MRBID
# o_imm  =       5  # Immediate Value -> IV
# o_far  =       6  # Immediate Far Address  (CODE) -> IFA
# o_near  =      7  # Immediate Near Address (CODE) -> INA


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
            # opnd1 = changeType(inst['optype1'])
            # opnd2 = changeType(inst['optype2'])
            # opnd3 = changeType(inst['optype3'])

            # word = opcode.replace(" ", "") + opnd1.replace(" ", "") + opnd2.replace(" ", "") + opnd3.replace(" ", "")
            # word = opcode.replace(" ", "") + opnd1.strip() + opnd2.strip() + opnd3.strip()
            word = opcode + opnd1 + opnd2 + opnd3
            # print(word)
            word = word.strip()
            word_list.append(word)
    return word_list


result_dict={}
count = 1
for bb_range_key in bb_range_list.keys():
    line_list = []
    for bb_data in bb_data_list:
        if bb_data['filename'] == bb_range_key:
            for bb_range in bb_range_list[bb_range_key]:
                # print(instlistbyline(bb_data['insts'], bb_range))
                try:
                    line_list.append(instlistbyline(bb_data['insts'], bb_range))
                except:
                    # pass
                    print('error', bb_data['filename'])
    result_dict[bb_range_key] = line_list
    print('result', count, 'complete', bb_range_key)
    count += 1

print('=> end result')

output_path = 'E:/Works/Data2/apt_lastoutput_original'

count = 1
for result_dict_key in result_dict.keys():
    try:
        output_file = os.path.join(output_path, result_dict_key)
        of = open(output_file, 'w')
        resultlist = []
        for line in result_dict[result_dict_key]:
            oneline = " ".join(line)
            resultlist.append(oneline)
        of.write("\n".join(resultlist))
        of.close()
        print('write', count, 'compelte', result_dict_key)
        count += 1
    except:
        print('write', count, 'error', result_dict_key)
        count += 1

print('=> file write end')
# print('===================> count :', count)
#
# tcount = tcount + 1
# print('!!!!!!!!!!!!!!!tcount!!!!!!!!!!!', tcount)