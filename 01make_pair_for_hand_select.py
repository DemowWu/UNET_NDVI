import tkinter
from PIL import Image, ImageTk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter.messagebox as ms

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

dirpath = '/media/hb/My Passport/unetcut2'
outdir = '/media/hb/My Passport/unetchoose'
ccc = len(os.listdir('/media/hb/My Passport/unetcut')) + 10
n = len(os.listdir(dirpath)) // 2
for i in range(ccc, ccc + n):
    a = np.load(os.path.join(dirpath, "{}.npy".format(i)))
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.imshow(a, cmap='PuBuGn')
    a = np.load(os.path.join(dirpath, "{}_mask.npy".format(i)))
    ax3 = fig.add_subplot(1, 2, 2)
    plt.imshow(a, cmap='PuBuGn')
    plt.savefig(os.path.join(outdir, "{}.jpg".format(i)))
    plt.close()
    print(i)
