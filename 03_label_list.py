'''
Create bin2vec_dadtaset.zip
Create mann_dataset.zip
'''


# This code is to extract sample list over 20
# and rezip -> output/dataset.zip

import csv
import os
import zipfile


# read all samplefiles (_each_total.zip)
current_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = 'output'
file_path = os.path.join(current_directory, data_directory)
samples_zipfile = os.path.join(file_path, '_each_total.zip')
bin2vec_dataset_filename = 'bin2vec_dataset.zip'
mann_dataset_filename = 'mann_dataset.zip'
temp_folder = 'E:/Works/temp/samples/'



def get_filelist(zip_path):
    with zipfile.ZipFile(zip_path) as fi:
        flist = [c for c in fi.namelist()]

    return flist

# 0 byte file delete
# fl = os.listdir(temp_folder)
#
# for f in fl:
#     file = temp_folder + f
#     n = os.path.getsize(file)
#     if n == 0:
#         os.remove(file)

# print(len(get_filelist(samples_zipfile)))

filel = get_filelist(samples_zipfile)

f = open('temp/label.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
count = 0


# ts and asd same detect name -> tsasd_list
tsasd_list = list()
ntsasd_list = list()
for line in rdr:

    ts = line[2]
    asd = line[3]

    if ts == asd:
        tsasd_list.append(line)
    else:
        ntsasd_list.append(line)

f.close()
print('tsasd_list', len(tsasd_list))
print('ntasd_list', len(ntsasd_list))
# print(tsasd_list)



# re_tsasd_list
re_tsasd_list = list()
nre_tsasd_list = list()
for tsasd in tsasd_list:
    md5 = tsasd[0].strip()
    if md5 in filel:
        re_tsasd_list.append(tsasd)
    else:
        nre_tsasd_list.append(tsasd)
#
print('re_tsasd_list', len(re_tsasd_list))
print('nre_tsasd_list', len(nre_tsasd_list))
# print(len(ntsasd_list))


# make dictionary by asd detect name -> asdic
asdic = dict()
doc2veclist = list()
for li in re_tsasd_list:
    asd = li[3].strip()
    result = asdic.get(asd)

    if result == None:
        templist = list()
        templist.append(li[0].strip())
        asdic[asd] = templist

    else:
        asdic[asd].append(li[0].strip())

for li in nre_tsasd_list:
        doc2veclist.append(li[0].strip())

for li in ntsasd_list:
        doc2veclist.append(li[0].strip())

temp = 0
for key in asdic.keys():
    vl = asdic[key]
    count = len(vl)
    temp = temp + count

print('asdic_len', temp)
# print(len(asdic.keys()))
#
print('doc2veclen', len(doc2veclist))
# print(len(asdic))
# print(len(nasdic))



# make dictionary over 20 samples -> over20, etc
over20 = dict()
etc = dict()
for key in asdic.keys():
    value_list = asdic[key]
    if len(value_list) > 20:
        over20[key] = value_list[:20]
        etc[key] = value_list[20:]
    else:
        etc[key] = value_list

print('over20 keys', len(over20))
print('etc keys', len(etc))
total = 0
for key in over20.keys():
    vl = over20[key]
    count = len(vl)
    total = total + count

print('over20', total)

for key in etc.keys():
    value_list = etc[key]
    doc2veclist.extend(value_list)

print('re doc2veclist', len(doc2veclist))

# delete duplication
doc2veclist = list(set(doc2veclist))
print('no duplication doc2veclist', len(doc2veclist))


# extract_zip
def extract(zipfilepath, extract_folder):
    try:
        ext_zip = zipfile.ZipFile(zipfilepath)
        ext_zip.extractall(extract_folder)

        ext_zip.close()
        return 0
    except:
        print('extract zipfile error')
        return 1


def write_zip(output_file, file_folder, file_list):

        zf = zipfile.ZipFile(output_file, 'w')
        for file in file_list:
            try:
                zf.write(os.path.join(file_folder, file), file, compress_type=zipfile.ZIP_DEFLATED)
            except:
                print('write zip error' + output_file)
        zf.close()

# result = extract(samples_zipfile, temp_folder)
#
# if result == 0:
#     m_file_list = list()
#     for key in over20.keys():
#         md5list = over20[key]
#         for md5 in md5list:
#             m_file_list.append(md5.strip())
#
#     write_zip(file_path+'/'+mann_dataset_filename, temp_folder, m_file_list)

write_zip(file_path+'/'+bin2vec_dataset_filename, temp_folder, doc2veclist)





























# #
# def read_zip_file_list(samples_zipfile):
#     with zipfile.ZipFile(samples_zipfile) as f:
#         binList = [hash for hash in f.namelist()]
#         for bin in binList:
#             buf = str()
#             for idx, line in enumerate(f.open(bin)):
#                 buf = buf + str(line.strip().decode('utf-8'))
#             bfLabel = bin
#             wl = bfLabel + ' ' + buf + '\n'
#             of.write(wl)
#             yield TaggedDocument(gensim.utils.simple_preprocess(buf.strip(), min_len=1, max_len=100), [bfLabel])
#             print(bin, 'read, write completed')
#     of.close()