import json
import os

data_directory = 'E:/Works/Data/samples/output/'
output_directory = 'E:/Works/Data/samples/output_c/'
output_filename = 'mysearch7'

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

print('basicblocklist')
count = 1
for bblist_filename in bblist_filenamelist:
    bblist_data_file = open(bblist_filename, encoding='utf-8')
    bblist_data = json.load(bblist_data_file)
    bblist_data_list.append(bblist_data)
    print('basicblocklist', count, 'complete', bb_filenamelist)
    count += 1

print('=> json object read end')
print('=> make bblist line list file')

def search_line(filename, startaddress):
    for bb_data in bb_data_list:
        if filename == bb_data['filename']:
            for inst in bb_data['insts']:
                addr = inst['addr']
                line = inst['line']
                if startaddress == addr:
                    return line


bblist_data_start_line_list=[]

for bblist_data in bblist_data_list:
    bb_start_line_list = []
    filename = bblist_data['filename']
    for bblist in bblist_data['bblist']:
        startaddress = bblist['start']
        line = search_line(filename, startaddress)
        if line != -1:
            bb_start_line_list.append(line)

    bblist_data_start_line_list.append(bb_start_line_list)

# print(bblist_data_start_line_list)

print('=> search value make start')

word_list = []
bbcount = 1


# for bb_data in bb_data_list:
#     for inst in bb_data['insts']:
#         # opcode = inst['opcode']
#         # opnd1 = inst['opnd1']
#         # opnd2 = inst['opnd2']
#         # opnd3 = inst['opnd3']
#         # optype1 = inst['optype1']
#         # optype2 = inst['optype2']
#         # optype3 = inst['optype3']
#         opnd1 = inst['opnd1']
#         opnd2 = inst['opnd2']
#         opnd3 = inst['opnd3']
#         optype1 = inst['optype1']
#         optype2 = inst['optype2']
#         optype3 = inst['optype3']
#         # word = opcode.strip()+opnd1.strip()+opnd2.strip()+opnd3.strip()
#         # word = word.strip()
#         # word_list.append(word)
#         searchvalue = 7
#
#         if (optype1 == searchvalue):
#             word_list.append(opnd1)
#         elif (optype2 == searchvalue):
#                 word_list.append(opnd2)
#         elif (optype3 == searchvalue):
#             word_list.append(opnd3)
#
#     print(bbcount, 'complete', bb_data['filename'])
#     bbcount += 1
#
# print('=> search value make end')
# print('=> search value make start')
#
# output_file = open(output_directory + output_filename, 'w')
# output_file.write(" ".join(word_list))
# output_file.close()
#
# print('word_list size', len(word_list))
# ndword_list = list(set(word_list))
# print('not dupl word_list size', len(ndword_list))
#
#
# print('=> search value make end')

# GetOPType
# o_void  =      0  # No Operand
# o_reg  =       1  # General Register (al,ax,es,ds...)    reg
# o_mem  =       2  # Direct Memory Reference  (DATA)      addr
# o_phrase  =    3  # Memory Ref [Base Reg + Index Reg]    phrase
# o_displ  =     4  # Memory Reg [Base Reg + Index Reg + Displacement] phrase+addr
# o_imm  =       5  # Immediate Value                      value
# o_far  =       6  # Immediate Far Address  (CODE)        addr
# o_near  =      7  # Immediate Near Address (CODE)        addr
# 8: I don't remember what... something related to MOVs

# GetOperandValue
# operand is an immediate value  => immediate value
# operand has a displacement     => displacement
# operand is a direct memory ref => memory address
# operand is a register          => register number
# operand is a register phrase   => phrase number
# otherwise                      => -1

# json sample
# "comment": "",
# "opndv3": -1,
# "opndv2": 1,
# "opndv1": 4286700,
# "disasm": "sub     ds:dword_4168EC, 1",
# "opnd1": "ds:dword_4168EC",
# "line": 0,
# "optype2": 5,
# "opnd3": "",
# "addr": "0x00411000",
# "opcode": "sub",
# "fname": "sub_411000",
# "optype1": 2,
# "opnd2": "1",
# "optype3": 0