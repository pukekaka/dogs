import pefile
import os

# file_directory = 'E:/Works/Data/samples/malwares/APT/'
# file_list = []
# count = 1

# for root, dirs, files in os.walk(file_directory):
#     for fn in files:
#         idbfile = fn.split('.')
#         # print(idbfile)
#         if len(idbfile) > 1 :
#             if idbfile[1] == 'til' or 'nam' or 'i*':
#                 # print(idbfile)
#                 path = root + fn
#                 os.remove(path)

filename = 'E:/Works/Data/samples/output_c/basicblock_by_line'
output = 'E:/Works/Data/samples/output_c/basicblock_by_space'

templist = []
f = open(filename, 'r')
lines = f.readlines()
for i, line in enumerate(lines):
    sline = line.split(' ')
    templist = templist + sline
    print(i, len(templist), 'complete')
    # print(templist)
    # data = templist

f = open(output, 'w')
f.write(str(templist))
f.close()

#
# for root, dirs, files in os.walk(file_directory):
#     for file in files:
#         try:
#             pe = pefile.PE(root + file)
#             machine_bit = pe.FILE_HEADER.Machine
#             if machine_bit == 332:
#                 test = 'haha'
#             else:
#                 test = 'nono'
#             print(count, file, test)
#         except:
#             print(count, file, 'error')
#
#         count += 1
#         # pe = pefile.PE('E:/Works/Data/samples/malwares/APT/030d395e084330c91e383ff3e9fc4e78')
#         # file_list.append(file)






# pe = pefile.PE('E:/Works/Data/samples/malwares/APT/030d395e084330c91e383ff3e9fc4e78')
# machine_bit = pe.FILE_HEADER.Machine
# print(machine_bit)


# if machine_bit == 0x14c :
#     print ("x86")
# elif machine_bit == 0x200 :
#     print ("x64")


#define IMAGE_FILE_MACHINE_UNKNOWN               0
#define IMAGE_FILE_MACHINE_I386                  0x014c  // Intel 386.
#define IMAGE_FILE_MACHINE_R3000               0x0162  // MIPS little-endian, 0x160 big-endian
#define IMAGE_FILE_MACHINE_R4000               0x0166  // MIPS little-endian
#define IMAGE_FILE_MACHINE_R10000             0x0168  // MIPS little-endian
#define IMAGE_FILE_MACHINE_WCEMIPSV2       0x0169  // MIPS little-endian WCE v2
#define IMAGE_FILE_MACHINE_ALPHA             0x0184  // Alpha_AXP
#define IMAGE_FILE_MACHINE_SH3                 0x01a2  // SH3 little-endian
#define IMAGE_FILE_MACHINE_SH3DSP             0x01a3
#define IMAGE_FILE_MACHINE_SH3E               0x01a4  // SH3E little-endian
#define IMAGE_FILE_MACHINE_SH4                0x01a6  // SH4 little-endian
#define IMAGE_FILE_MACHINE_SH5                0x01a8  // SH5
#define IMAGE_FILE_MACHINE_ARM                0x01c0  // ARM Little-Endian
#define IMAGE_FILE_MACHINE_THUMB             0x01c2
#define IMAGE_FILE_MACHINE_AM33              0x01d3
#define IMAGE_FILE_MACHINE_POWERPC           0x01F0  // IBM PowerPC Little-Endian
#define IMAGE_FILE_MACHINE_POWERPCFP         0x01f1
#define IMAGE_FILE_MACHINE_IA64                   0x0200  // Intel 64
#define IMAGE_FILE_MACHINE_MIPS16                0x0266  // MIPS
#define IMAGE_FILE_MACHINE_ALPHA64           0x0284  // ALPHA64
#define IMAGE_FILE_MACHINE_MIPSFPU           0x0366  // MIPS
#define IMAGE_FILE_MACHINE_MIPSFPU16         0x0466  // MIPS
#define IMAGE_FILE_MACHINE_AXP64             IMAGE_FILE_MACHINE_ALPHA64
#define IMAGE_FILE_MACHINE_TRICORE           0x0520  // Infineon
#define IMAGE_FILE_MACHINE_CEF               0x0CEF
#define IMAGE_FILE_MACHINE_EBC               0x0EBC  // EFI Byte Code
#define IMAGE_FILE_MACHINE_AMD64             0x8664  // AMD64 (K8)
#define IMAGE_FILE_MACHINE_M32R              0x9041  // M32R little-endian
#define IMAGE_FILE_MACHINE_CEE               0xC0EE
