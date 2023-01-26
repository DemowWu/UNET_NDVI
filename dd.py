import matplotlib.pyplot as plt
import os
import numpy as np
# n = len(os.listdir('/home/hb/work/unetcut')) // 2
import datetime
import pickle as pkl

# parse = argparse.ArgumentParser()
# parse.add_argument("--action", type=str, help="train/test/train&test", default="train")  # default="train&test"
# parse.add_argument('--arch', '-a', metavar='ARCH', default='UNet1', help='UNet/UNet1/UnetD/UNetVGG/VAE/resnet34_unet/unet++/myChannelUnet/Attention_UNet/segnet/r2unet/fcn32s/fcn8s')
#
# # parse.add_argument("--data_dir", default='/public/home/xuezhe/unetselected')
# parse.add_argument("--data_dir", default='/home/hb/work/unetcut')
# parse.add_argument("--checkpoints", default='result/model', help="the path of model weight file")
# parse.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
# parse.add_argument("--log_dir", default='result/log', help="log dir")
# parse.add_argument('--log_step', type=int, default=15, help='log intervals')
# parse.add_argument('--val_step', type=int, default=1, help='validation_interval')
#
# parse.add_argument("--latent_variable_size", type=int, default=500)
# parse.add_argument("--rec_loss", type=str, default="mse", choices=("ssim", "bce", "l1", "spl", 'mse'))
# parse.add_argument("--fmaps", type=int, default=64, help="features maps channels")
# parse.add_argument("--nc", type=int, default=1, help="input image channels")
#
# parse.add_argument("--shuffle", type=bool, default=True)  # shuffle the data set
# parse.add_argument("--pin_memory", type=bool, default=True)  # use pinned (page-locked) memory. when using CUDA, set to True
# parse.add_argument('--num_workers', default=1)  # number of threads for data loading
#
# parse.add_argument("--epoch", type=int, default=50)
# parse.add_argument("--accumulation_steps", type=int, default=3)
# parse.add_argument("--batch_size", type=int, default=18)
# parse.add_argument("--one_batch", type=int, default=6)  # batch_size/subdivide  e.g. 12/2=6
#
# parse.add_argument('--learning_rate', default=1e-3)
# parse.add_argument('--weight_decay', default=1e-4)
#
# args = parse.parse_args()
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

trains, vals, tests = pkl.load(open('data/unet_trainset1.pkl', 'rb'))
kk=0
for f in trains:
    a=np.load('/home/hb/work/unetselected/{}.npy'.format(f))
    if 1<np.max(a) or -1>np.min((a)):
        kk+=1
        print(kk,np.max(a),np.min(a))
        a=normalization(a)
        plt.imshow(a, cmap='PuBuGn')
        plt.show()
        n = 60000
logfile = '/home/hb/work/xuezhe/ndvi/UNET-ZOO/result/sdf.log'
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

print(n, len(kk), len(index_list))
