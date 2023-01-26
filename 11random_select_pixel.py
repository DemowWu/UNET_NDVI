import random
import pickle as pkl
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_source_data_distribute():
    # trains, vals, tests = pkl.load(open('data/unet_trainset1.pkl', 'rb'))
    files = os.listdir('/home/hb/work/tmp/unetouttest/test')

    data_dir = '/home/hb/work/unetcut'
    all_cord_for_draw = []
    for i in range(len(files)):
        tests = files[i].split('_')[0]
        cord = [[random.randint(0, 511), random.randint(0, 511)] for _ in range(5000)]
        tile = np.load(os.path.join(data_dir, '{}.npy'.format(tests)))
        tile_mask = np.load(os.path.join(data_dir, '{}_mask.npy'.format(tests)))
        tile_predict = np.load(os.path.join('/home/hb/work/tmp/unetouttest/test', '{}_mask.npy'.format(tests)))
        for j in range(500):
            x = cord[j][0]
            y = cord[j][1]
            all_cord_for_draw.append([tile[x, y], tile_predict[x, y]])
    cords = np.array(all_cord_for_draw)
    fig = plt.figure(figsize=(18, 6))
    # ax1 = fig.add_subplot(1, 2, 1)
    # plt.scatter(cords[:,0],cords[:,1],s=1)
    # ax2 = fig.add_subplot(2, 2, 1)
    plt.scatter(cords[:, 0], cords[:, 1], s=1)
    plt.show()
    print(cords.shape[0], cords.shape[1])



