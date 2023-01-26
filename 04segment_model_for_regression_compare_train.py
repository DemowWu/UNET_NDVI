import logging
import argparse
import os
import shutil
import time
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from segment_models.models.DeepLabV3 import deeplabv3
from segment_models.models.DeepLabV3_plus import deeplabv3_plus
from segment_models.models.PSPNet import pspnet
from segment_models.models.Unet_AE import unet_ae
from segment_models.models.Unet import Unet2
from segment_models.models.Segnet import segnet
from segment_models.models.CENet import cenet
from segment_models.models.Unet_nested import UNet_Nested
from segment_models.models.DenseASPP import denseaspp
from segment_models.models.RefineNet import RefineNet
from segment_models.models.RDFNet import rdfnet
import math
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from ztools import *
import yaml
from attrdict import AttrMap
from ndvi_loss.ssim_loss import SSIMLoss
from ndvi_loss.mssim_l1 import MS_SSIM_L1_LOSS
from ndvi_loss.mssim_l1_1 import MS_SSIM_L1_LOSS2
# import skimage.metrics as measure
import skimage.measure as measure
import pickle as pkl
from torch.utils.data import DataLoader
from torch.autograd import Variable
from skimage.transform import resize

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def get_args():
    with open('config/att_damodule_unet1.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    args = AttrMap(config)
    return args


def get_log(args, filen):
    dirname = os.path.join(args.log_dir, 'segment')
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = dirname + '/' + filen + '.log'
    logging.basicConfig(filename=filename, level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
    return logging


def get_model(args):
    model = None
    m_name = 'UNet'
    if args == '1':
        model = deeplabv3.DeepLabV3(class_num=1)
        m_name = 'deeplabv3'
    elif args == '2':
        model = deeplabv3_plus.DeepLabv3_plus(in_channels=1, num_classes=1, backend='mobilenet_v2', os=16, pretrained='imagenet')
        m_name = 'deeplabv3_plus'
    elif args == '3':
        model = pspnet.PSPNet(in_channels=1, num_classes=1, backend='resnet101', pool_scales=(1, 2, 3, 6), pretrained='imagenet')
        m_name = 'pspnet'
    elif args == '4':
        model = unet_ae.UnetResnetAE(in_channels=1, num_classes=1, backend='resnet101', pretrained='imagenet')
        m_name = 'unet_ae'
    elif args == '5':
        model = Unet2.U_Net(img_ch=1, output_ch=1)
        m_name = 'U_Net'
    elif args == '6':
        model = Unet2.R2U_Net(img_ch=1, output_ch=1, t=2)
        m_name = 'R2U_Net'
    elif args == '7':
        model = Unet2.AttU_Net(img_ch=1, output_ch=1)
        m_name = 'Attunet'
    elif args == '8':
        model = Unet2.R2AttU_Net(img_ch=1, output_ch=1, t=2)
        m_name = 'R2Attunet'
    elif args == '9':
        model = segnet.SegNet(num_classes=1, in_channels=1)
        m_name = 'segnet'
    elif args == '10':
        model = cenet.CE_Net(num_classes=1, num_channels=1)
        m_name = 'cenet'
    elif args == '11':
        model = UNet_Nested.UNet_Nested(in_channels=1, n_classes=1)
        m_name = 'unet_nested'
    elif args == '12':
        model = denseaspp.DenseASPP(class_num=1)
        m_name = 'denseaspp'
    elif args == '13':
        model = RefineNet.get_refinenet(input_size=256, num_classes=1)
        m_name = 'refinenet'

    return model, m_name


def get_dataset(args, dataset):
    if dataset == '1':
        trains, vals, tests = pkl.load(open('data/unet_trainset1.pkl', 'rb'))
    if dataset == '2':
        trains, vals, tests = pkl.load(open('data/unet_trainset2.pkl', 'rb'))
    if dataset == '3':
        trains, vals, tests = pkl.load(open('data/unet_trainset3.pkl', 'rb'))
    if dataset == '4':
        trains, vals, tests = pkl.load(open('data/unet_trainset4.pkl', 'rb'))
    tr_data = NdviDataset(args.data_dir, trains, False, transform=x_transforms, target_transform=y_transforms)
    tr_loader = DataLoader(tr_data, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
    val_data = NdviDataset(args.data_dir, vals, False, transform=x_transforms, target_transform=y_transforms)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = val_loader
    return tr_loader, val_loader, test_loader


def train(model, train_loader, criterion, epoch, args):
    model.train()
    num_batches = 0
    avg_loss = []
    dt_size = len(train_loader.dataset)
    for x, y, _, mask in train_loader:
        if x.shape[0] > 4:
            inputs = x.cuda()
            labels = y.cuda()
            optimizer.zero_grad()  # 梯度清零
            output = model(inputs)
            loss = criterion(output, labels)
            avg_loss.append(loss.item())
            loss.backward()
            optimizer.step()  # 更新参数
            num_batches += 1

            if num_batches % args.log_step == 0:
                print("%d/%d,loss:%0.6f" % (num_batches, (dt_size - 1) // train_loader.batch_size + 1, loss.item()))
                logging.info("%d/%d,loss:%0.6f" % (num_batches, (dt_size - 1) // train_loader.batch_size + 1, loss.item()))

    print('epoch: {} train_loss1:{} train_loss2:{}'.format(epoch, np.mean(avg_loss), 0))
    logging.info('epoch: {} train_loss1:{} train_loss2:{}'.format(epoch, np.mean(avg_loss), 0))


def get_metrics(image_mask, predict):
    mse = measure.compare_mse(image_mask, predict)
    psnr = measure.compare_psnr(image_mask, predict, data_range=2)
    # ssim = measure.compare_ssim(image_mask, predict)
    # mse = measure.mean_squared_error(image_mask, predict)
    # psnr = measure.peak_signal_noise_ratio(image_mask, predict, data_range=2)
    # ssim = measure.structural_similarity(image_mask, predict, data_range=2,multichannel=True)
    rmse = math.sqrt(mse)
    return rmse, psnr  # , ssim


def val(model, val_loader, criterion, epoch):
    model.eval()
    num_batches = 0
    losses = []

    rmse_sum = []
    psnr_sum = []
    ssim_sum = []

    with torch.no_grad():
        for x, y, _, mask in val_loader:
            if x.shape[0] > 4:
                inputs = x.cuda()
                labels = y.cuda()
                output = model(inputs)
                loss = criterion(output, labels)
                losses.append(loss.item())

                rmse, psnr = get_metrics(labels.detach().cpu().numpy(), output.detach().cpu().numpy())
                rmse_sum.append(rmse)
                psnr_sum.append(psnr)
                # ssim_sum.append(ssim)

                num_batches += 1

    print('epoch: ' + str(epoch) + ', val_loss: ' + str(np.mean(losses)))
    logging.info('epoch: ' + str(epoch) + ', val_loss: ' + str(np.mean(losses)))
    logging.info('epoch: ' + str(epoch) + ', rmse: ' + str(np.mean(rmse_sum)))
    logging.info('epoch: ' + str(epoch) + ', psnr: ' + str(np.mean(psnr_sum)))
    # logging.info('epoch: ' + str(epoch) + ', ssim: ' + str(np.mean(ssim_sum)))
    return np.mean(losses)


def run_train_val(model, lr_scheduler, best_loss, start_epoch, tr_loader, val_loader, args, criterion, file_name):
    for epoch in range(start_epoch, args.epoch):
        train(model, tr_loader, criterion, epoch, args)
        if epoch % args.val_step == 0:
            val_loss = val(model, val_loader, criterion, epoch)
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = os.path.join(args.checkpoints, file_name + ".pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "best_loss": best_loss,
                        "optimizer": optimizer.state_dict(),
                    },
                    save_path,
                )
                print("Saved checkpoint to: %s" % save_path)

        lr_scheduler.step()


def run_train(model, tr_loader, opt, criterion):
    for epoch in range(1, opt.epochs):
        train(model, tr_loader, criterion, epoch)


def run_test(model, test_loader, opt):
    model.eval()
    predictions = []
    img_ids = []
    for batch_idx, sample_batched in enumerate(test_loader):
        data, img_id, height, width = sample_batched['image'], sample_batched['img_id'], sample_batched['height'], sample_batched['width']
        data = Variable(data.type(opt.dtype))
        output = model.forward(data)
        # output = (output > 0.5)
        output = output.data.cpu().numpy()
        output = output.transpose((0, 2, 3, 1))  # transpose to (B,H,W,C)
        for i in range(0, output.shape[0]):
            pred_mask = np.squeeze(output[i])
            id = img_id[i]
            h = height[i]
            w = width[i]
            # in p219 the w and h above is int
            # in local the w and h above is LongTensor
            if not isinstance(h, int):
                h = h.cpu().numpy()
                w = w.cpu().numpy()
            pred_mask = resize(pred_mask, (h, w), mode='constant')
            pred_mask = (pred_mask > 0.5)
            predictions.append(pred_mask)
            img_ids.append(id)

    return predictions, img_ids


if __name__ == '__main__':
    x_transforms = transforms.ToTensor()
    y_transforms = transforms.ToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    print('model name')
    model_order = input()

    print('loss function(mse/ssim/mssim_l1/mssim_l11)')
    args.rec_loss = input()

    print('additional information')
    model_alias = input()

    print('batch size')
    args.batch_size = int(input())

    # print('scheduler name(step/expo)')
    scheduler_name = 'expo'  # input()

    # print('dataset number(1/2/3/4)')
    # dataset_number = input()
    dataset_number = '2'

    print('learning rate')
    args.learning_rate = float(input())

    print('weight decay')
    args.weight_decay = float(input())

    model, model_name = get_model(model_order)
    args.arch = model_name
    model = nn.DataParallel(model) # 并行

    file_name = model_name + '_' + args.rec_loss + '_data' + str(dataset_number) + '_e' + str(args.epoch) + '_b' + str(args.batch_size) + '_' + model_alias
    logging = get_log(args, file_name)
    logging.info('model name:' + model_order)
    logging.info('loss function:' + args.rec_loss)
    logging.info('additional information:' + model_alias)
    logging.info('batch size:' + str(args.batch_size))
    logging.info('scheduler name:' + scheduler_name)
    logging.info('dataset:' + dataset_number)
    logging.info('learning rate:' + str(args.learning_rate))
    logging.info('weight decay:' + str(args.weight_decay))
    print('models:%s,\nepoch:%s,\nbatch size:%s' % (args.arch, args.epoch, args.batch_size))
    print(file_name)
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\n========' % (args.arch, args.epoch, args.batch_size))

    train_loader, val_loader, test_loader = get_dataset(args, dataset_number)
    best_loss = 999
    start_epoch = 0

    if 'train' in args.action:
        model = model.cuda()

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        # loss function
        criterion = None
        if args.rec_loss == 'mse':
            criterion = nn.MSELoss().cuda()
        if args.rec_loss == 'ssim':
            criterion = SSIMLoss().cuda()
        if args.rec_loss == 'mssim_l1':
            criterion = MS_SSIM_L1_LOSS().cuda()
        if args.rec_loss == 'mssim_l11':
            criterion = MS_SSIM_L1_LOSS2().cuda()

        # scheduler
        lr_scheduler = None
        if scheduler_name == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        if scheduler_name == 'expo':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

        # resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                start_epoch = checkpoint["epoch"]
                best_loss = checkpoint["best_loss"]
                model.load_state_dict(checkpoint["state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        run_train_val(model, lr_scheduler, best_loss, start_epoch, train_loader, val_loader, args, criterion, file_name)

    if 'test' in args.action:
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model-01.pt')))
        model = model.cuda()
        predictions, img_ids = run_test(model, test_loader, args)
