import tkinter
from PIL import Image, ImageTk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter.messagebox as ms
from skimage.measure import compare_ssim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from matplotlib.widgets import Button
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
dirpath = '/home/hb/work/unetcut'
outdir = '/media/hb/Elements/unetchoose2'
n = len(os.listdir(dirpath)) // 2
kk=0
for i in range(5000):
    a = np.load(os.path.join(dirpath, "{}.npy".format(i)))
    b = np.load(os.path.join(dirpath, "{}_mask.npy".format(i)))
    r2 = r2_score(a.flatten(), b.flatten())
    if r2>0.001:
        kk+=1
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(1, 2, 1)

        plt.text(800, -10, "R2 {}".format(r2), fontsize=12)  # 标题
        plt.imshow(a, cmap='PuBuGn')
        ax3 = fig.add_subplot(1, 2, 2)
        plt.imshow(b, cmap='PuBuGn')

        plt.savefig(os.path.join(outdir, "{}.jpg".format(i)))
        plt.close()
        # r3=compare_ssim(a,b,data_range=1)
        print(kk,i,r2)