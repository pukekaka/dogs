#!/usr/bin/python

'''

execute python 2.7 !!!!!

'''

import os
import pefile

def main():

    file_directory = 'E:/Works/Data/samples/malwares/APT/'
    script_directory = './idapython_script/'
    # bbext_asmb = script_directory + 'basicblock_extractor_asmb.py'
    bbext_json = script_directory + 'basicblock_extractor_json.py'
    bbext_bblist = script_directory + 'extractor_bblist.py'

    # binary list append
    file_pathlist = []
    file_namelist= []
    for root, dirs, files in os.walk(file_directory):
        for file in files:
            file_namelist.append(file)
            file_pathlist.append(file_directory+file)

    count = 1

    for file_path in file_pathlist:

        try:
            print (count, 'start : ', file_path)

            pe = pefile.PE(file_path)
            machine_bit = pe.FILE_HEADER.Machine
            # print(root+file)
            # print(machine_bit)
            # print(count, file, machine_bit)
            if machine_bit == 332:
                command = 'idaw'
                # print(command)
            else:
                command = 'idaw64'
                # print(command)

            # cmd = command+' -c -A -S' + bbext_json + ' ' + file_path+' > NUL'
            # print('     select command : ', cmd)
            #
            # result1 = os.system(cmd)
            # if result1 == 0:
            #     print('     mj_complete: %s' % file_path)
            # else:
            #     print('     mj_error: %s' % file_path)

            print('     => next step start')

            cmd2 = command + ' -c -A -S' + bbext_bblist + ' ' + file_path+' > NUL'
            print('     select command2 : ', cmd2)
            result2 = os.system(cmd2)
            if result2 == 0:
                print('     bblist_complete: %s' % file_path)
            else:
                print('     bblist_error: %s' % file_path)

            print('     => End all step')

        except:
            print(file, 'error')

        count += 1


    for root, dirs, files in os.walk(file_directory):
        for fn in files:
            idbfile = fn.split('.')
            # print(idbfile)
            if len(idbfile) > 1:
                if idbfile[1] == 'til' or 'nam' or 'i*':
                    # print(idbfile)
                    path = root + fn
                    os.remove(path)

if __name__ == "__main__":
	main()






    # for file_name2 in file_namelist:
    #     idbfile = file_name2.split('.')[0] + '.idb'
    #     os.remove(file_directory + idbfile)
    #
    # print(count, '!!!!!!!!!!!idb file ALL Delete!!!!!!!!!!!!!')