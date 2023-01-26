import argparse
import logging
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import autograd, optim, nn, squeeze,device,load
from ndvi_models.UNet import Unet, resnet34_unet, UnetDense
from ndvi_models.attention_unet import AttU_Net
from ndvi_models.channel_unet import myChannelUnet
from ndvi_models.r2unet import R2U_Net
from ndvi_models.segnet import SegNet
from ndvi_models.unetpp import NestedUNet
from ndvi_models.fcn import get_fcn8s
from ndvi_models.cenet import CE_Net_
from ztools import *
from torchvision.transforms import transforms
import pickle as pkl
def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0)
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train")  # default="train&test"
    parse.add_argument("--epoch", type=int, default=60)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='UNet', help='UNet/UnetD/resnet34_unet/unet++/myChannelUnet/Attention_UNet/segnet/r2unet/fcn32s/fcn8s')
    parse.add_argument("--batch_size", type=int, default=8)
    # parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--data_dir", default='/home/hb/work/unetcut')
    parse.add_argument("--threshold", type=float, default=None)
    args = parse.parse_args()
    return args

def get_selected_files(args):
    # n = len(os.listdir(args.data_dir)) // 2
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

    return index_list

def train_val_test_split(args):
    # all files but not selected by hand
    # n = len(os.listdir(args.data_dir)) // 2  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
    # print(n)
    # index_list = list(range(n))
    # random.shuffle(index_list)
    # trains = index_list[0:int(n * 0.3)]
    # vals = index_list[int(n * 0.3):int(n * 0.4)]
    # tests = index_list[int(n * 0.4):]

    # get files selected by hand
    index_list = get_selected_files(args)
    n = len(index_list)
    random.shuffle(index_list)
    trains = index_list[0:int(n * 0.6)]
    vals = index_list[int(n * 0.6):int(n * 0.8)]
    tests = index_list[int(n * 0.8):]
    return trains, vals, tests

def get_model(args):
    if args.arch == 'UNet':
        model = Unet(1, 1).to(device)
    if args.arch == 'UNetD':
        model = UnetDense(1, 1).to(device)
    if args.arch == 'resnet34_unet':
        model = resnet34_unet(1, pretrained=False).to(device)
    if args.arch == 'unet++':
        args.deepsupervision = True
        model = NestedUNet(args, 3, 1).to(device)
    if args.arch == 'Attention_UNet':
        model = AttU_Net(1, 1).to(device)
    if args.arch == 'segnet':
        model = SegNet(3, 1).to(device)
    if args.arch == 'r2unet':
        model = R2U_Net(3, 1).to(device)
    # if args.arch == 'fcn32s':
    #     model = get_fcn32s(1).to(device)
    if args.arch == 'myChannelUnet':
        model = myChannelUnet(3, 1).to(device)
    if args.arch == 'fcn8s':
        assert args.dataset != 'esophagus', "fcn8s模型不能用于数据集esophagus，因为esophagus数据集为80x80，经过5次的2倍降采样后剩下2.5x2.5，分辨率不能为小数，建议把数据集resize成更高的分辨率再用于fcn"
        model = get_fcn8s(1).to(device)
    if args.arch == 'cenet':

        model = CE_Net_().to(device)

    return model


def get_dataset(args):
    trains, vals, tests = pkl.load(open('data/unet_trainset1.pkl', 'rb'))
    tr_data = NdviDataset(args.data_dir, trains, transform=x_transforms, target_transform=y_transforms)
    tr_loader = DataLoader(tr_data, batch_size=args.batch_size)
    val_data = NdviDataset(args.data_dir, vals, transform=x_transforms, target_transform=y_transforms)
    val_loader = DataLoader(val_data, batch_size=1)
    test_loader = val_loader
    return tr_loader, val_loader, test_loader




if __name__ == "__main__":
    our_dir = '/home/hb/work/tmp/unetouttest'
    x_transforms = transforms.ToTensor()
    y_transforms = transforms.ToTensor()
    device = device("cpu")
    args = get_args()
    print('models:%s,\nepoch:%s,\nbatch size:%s' % (args.arch, args.epoch, args.batch_size))
    model = get_model(args)
    train_loader, val_loader, test_loader = get_dataset(args)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    model.load_state_dict(load('/home/hb/work/pywork/ndvi2/UNET-ZOO/result/saved_model/UNet_4_150_0.006106630390723577.pth', map_location='cpu'))  # 载入训练好的模型
    model.eval()

    mse_sum = 0
    mae_sum = 0
    r2_sum = 0
    ii = 0
    for pic, _, pic_path, mask in test_loader:
        pic = pic.to(device)
        predict = model(pic)
        img_y = squeeze(predict).cpu().detach().numpy()

        mse, mae, r2, loss = get_metrics(mask[0], img_y, None)
        mse_sum += mse
        mae_sum += mae
        r2_sum += r2
        ii += 1
        if ii > 30:
            break

        # print(mask[0].split('/')[5])
        # pkl.dump(img_y,open('/home/hb/work/tmp/unetouttest/test/{}'.format(mask[0].split('/')[5]),'wb'))

        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(1, 3, 1)
        plt.text(800, -80, "R2 {}".format(r2), fontsize=12)  # 标题

        picname = pic_path[0].split('/')
        picname = picname[-1].split('.')
        ax1.set_title(picname[0])
        plt.imshow(np.load(pic_path[0]), cmap='PuBuGn')
        # plt.show()
        ax3 = fig.add_subplot(1, 3, 2)
        mask1 = mask[0].split('/')
        mask1 = mask1[-1].split('.')
        ax3.set_title(mask1[0])
        plt.imshow(np.load(mask[0]), cmap='PuBuGn')

        ax2 = fig.add_subplot(1, 3, 3)
        ax2.set_title('predict')
        plt.imshow(img_y, cmap='PuBuGn')
        plt.savefig(os.path.join(our_dir, "{}.jpg".format(picname[0])))
        # plt.close()
        # plt.show()
