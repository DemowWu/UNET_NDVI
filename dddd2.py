from shutil import copyfile
import random
import pickle as pkl
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage.measure as measure
import math


def get_metrics(image_mask, predict):
    mse = measure.compare_mse(image_mask, predict)
    psnr = measure.compare_psnr(image_mask, predict, data_range=2)
    ssim = measure.compare_ssim(image_mask, predict)
    rmse = math.sqrt(mse)
    return rmse, psnr, ssim


# dd = '/home/hb/work/unetselect'
# trains, vals, tests = pkl.load(open('data/unet_trainset1.pkl', 'rb'))
# for i in range(len(trains)):
#     a = np.load(os.path.join(dd, '{}.npy'.format(trains[i])))
#     # print('a min,max:',a.min(),a.max())
#     b = np.load(os.path.join(dd, '{}_mask.npy'.format(trains[i])))
#     # print('b min,max:', b.min(), b.max())
#     print(get_metrics(a, b))
# n = 0.0001
# aa = []
# for i in range(200):
#     n = n * 0.98
#     print(i,"%7.12f"%n)
#     aa.append(n)
#
# xx = list(range(len(aa)))
# plt.plot(xx, aa)
# plt.show()


def get_selected_files():
    n = os.listdir('/home/hb/work/unetcut2')
    print(n)
    # n = 60000
    logfile = '/home/hb/work/pywork/ndvi2/UNet_NDVI/data/unet_cloud_data_list2001.txt'
    with open(logfile, 'r') as f1:
        list1 = f1.readlines()
        tiles = []
        for l in list1:
            if 'INFO:' in l:
                ll = l.split('INFO:')
                tiles.append(int(ll[1].strip()))

    kk = set(tiles)
    index_list = []
    for i in n:
        if 'mask' not in i:
            if int(i.split('.')[0].strip()) not in kk:
                print(os.path.join('/home/hb/work/unetcut2',i),os.path.join('/home/hb/work/unetselect',i))
                print(os.path.join('/home/hb/work/unetcut2',i.split('.')[0].strip()+'_mask.npy'),os.path.join('/home/hb/work/unetselect',i.split('.')[0].strip()+'_mask.npy'))

                copyfile(os.path.join('/home/hb/work/unetcut2',i),os.path.join('/home/hb/work/unetselect',i))
                copyfile(os.path.join('/home/hb/work/unetcut2',i.split('.')[0].strip()+'_mask.npy'),os.path.join('/home/hb/work/unetselect',i.split('.')[0].strip()+'_mask.npy'))

    return index_list



get_selected_files()

