import random
import pickle as pkl
import numpy as np
import os
import matplotlib.pyplot as plt


def getnerate_random_test_point_locs():
    trains, vals, tests = pkl.load(open('data/unet_trainset2.pkl', 'rb'))
    all_cord_for_draw = {}
    for i in range(len(tests)):
        print(i)
        cord = [[random.randint(0, 511), random.randint(0, 511)] for _ in range(10000)]
        all_cord_for_draw[tests[i]] = cord

    pkl.dump(all_cord_for_draw, open('data/unet_test_random_points.pkl', 'wb'))


def plot_source_data_distribute():
    # trains, vals, tests = pkl.load(open('data/unet_trainset1.pkl', 'rb'))
    pred = pkl.load(open('/home/hb/work/pywork/ndvi2/UNet_NDVI/result/test/unet_test_pred_pair.pkl', 'rb'))
    source = pkl.load(open('/home/hb/work/pywork/ndvi2/UNet_NDVI/result/test/unet_test_source_pair.pkl', 'rb'))
    x = np.linspace(0, 1, 100)
    y = x

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    source = np.array(source)
    plt.scatter(source[:, 0], source[:, 1], s=1)
    plt.plot(x, y,'-r')
    ax3 = fig.add_subplot(1, 2, 2)

    pred = np.array(pred)
    plt.scatter(pred[:, 0], pred[:, 1], s=1)
    plt.plot(x, y,'-r')
    plt.show()




if __name__ == '__main__':
    # getnerate_random_test_point_locs()
    plot_source_data_distribute()
