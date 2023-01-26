import os
import numpy as np
import shutil

def get_selected_files():
    n = len(os.listdir('/home/hb/work/unetcut')) // 2
    print(n)
    # n = 60000
    logfile = '/home/hb/work/pywork/ndvi2/UNET-ZOO/data/unet_cloud_data_list.txt'
    with open(logfile, 'r') as f1:
        list1 = f1.readlines()
        tiles = []
        for l in list1:
            if 'INFO:' in l:
                ll = l.split('INFO:')
                tiles.append(int(ll[1].strip()))

    kk = set(tiles)
    index_list = []
    for i in range(n):
        if i not in kk:
            index_list.append(i)

    return index_list


if __name__ == '__main__':
    files=get_selected_files()
    data_dir='/home/hb/work/unetcut'
    dst_dir='/home/hb/work/unetselected'
    ii=0
    for f in files:
        ii+=1
        print(ii,f)
        shutil.copyfile(os.path.join(data_dir,'{}.npy'.format(f)),os.path.join(dst_dir,'{}.npy'.format(f)))
        shutil.copyfile(os.path.join(data_dir, '{}_mask.npy'.format(f)), os.path.join(dst_dir, '{}_mask.npy'.format(f)))
