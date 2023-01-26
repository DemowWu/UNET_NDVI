import argparse
import logging
import numpy as np
import os
import math
import pickle as pkl
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from ndvi_models.unet import UNet, UNet3_32, UNet4_32, UNet4_128, UNet5_64, UNet5_256, UNet6_512
from ndvi_models.attention_unet import AttU_Net
from ndvi_models.vae import VAE
from ndvi_models.unet_danet import Unet6_512_DAUNet1, Unet6_512_DAUNet2, Unet6_512_DAUNet3
from ndvi_models.unet_danet import Unet6_512_DAUNet4, Unet6_512_DAUNet5, Unet6_512_DAUNet6
from ndvi_models.unet_danet import Unet6_512_DAUNet7, Unet6_512_DAUNet8, Unet6_512_DAUNet9, Unet6_512_DAUNet10, \
    Unet6_512_DAUNet11, Unet6_512_DAUNet12
from ndvi_models.unet_cbam import Unet6_512_CBAM1, Unet6_512_CBAM2, Unet6_512_CBAM3, Unet6_512_CBAM4
from ndvi_models.unet_cbam import Unet6_512_CBAM5, Unet6_512_CBAM6, Unet6_512_CBAM7, Unet6_512_CBAM8
from ndvi_models.unet_cbam import Unet6_512_CBAM9, Unet6_512_CBAM10, Unet6_512_CBAM11, Unet6_512_CBAM12
from ndvi_models.unet_bam import Unet6_512_BAM1, Unet6_512_BAM2, Unet6_512_BAM3, Unet6_512_BAM4
from ndvi_models.unet_CoTAttention import Unet6_512_CoTAttention1, Unet6_512_CoTAttention2, Unet6_512_CoTAttention3, \
    Unet6_512_CoTAttention4
from ndvi_models.unet_CoTAttention import Unet6_512_CoTAttention5, Unet6_512_CoTAttention6, Unet6_512_CoTAttention7, \
    Unet6_512_CoTAttention8
from ndvi_models.unet_CoTAttention import Unet6_512_CoTAttention9, Unet6_512_CoTAttention10, Unet6_512_CoTAttention11, \
    Unet6_512_CoTAttention12
from ndvi_models.unet_polarizedselfattention import Unet6_512_PolarizedAttention1, Unet6_512_PolarizedAttention2, \
    Unet6_512_PolarizedAttention3, Unet6_512_PolarizedAttention4
from ndvi_models.unet_polarizedselfattention import Unet6_512_PolarizedAttention5, Unet6_512_PolarizedAttention6, \
    Unet6_512_PolarizedAttention7, Unet6_512_PolarizedAttention8
from ndvi_models.unet_polarizedselfattention import Unet6_512_PolarizedAttention9, Unet6_512_PolarizedAttention10, \
    Unet6_512_PolarizedAttention11, Unet6_512_PolarizedAttention12
from ndvi_models.unet_psa import Unet6_512_PSA1, Unet6_512_PSA2, Unet6_512_PSA3, Unet6_512_PSA4
from ndvi_models.unet_psa import Unet6_512_PSA5, Unet6_512_PSA6, Unet6_512_PSA7, Unet6_512_PSA8
from ndvi_models.unet_psa import Unet6_512_PSA9, Unet6_512_PSA10, Unet6_512_PSA11, Unet6_512_PSA12
from ndvi_models.unet_seattention import Unet6_512_SEAttention1, Unet6_512_SEAttention2, Unet6_512_SEAttention3, \
    Unet6_512_SEAttention4
from ndvi_models.unet_seattention import Unet6_512_SEAttention5, Unet6_512_SEAttention6, Unet6_512_SEAttention7, \
    Unet6_512_SEAttention8
from ndvi_models.unet_seattention import Unet6_512_SEAttention9, Unet6_512_SEAttention10, Unet6_512_SEAttention11, \
    Unet6_512_SEAttention12
from ndvi_models.unet_shuffleattention import Unet6_512_ShuffleAttention1, Unet6_512_ShuffleAttention2, \
    Unet6_512_ShuffleAttention3, Unet6_512_ShuffleAttention4
from ndvi_models.unet_shuffleattention import Unet6_512_ShuffleAttention5, Unet6_512_ShuffleAttention6, \
    Unet6_512_ShuffleAttention7, Unet6_512_ShuffleAttention8
from ndvi_models.unet_shuffleattention import Unet6_512_ShuffleAttention9, Unet6_512_ShuffleAttention10, \
    Unet6_512_ShuffleAttention11, Unet6_512_ShuffleAttention12
from ndvi_models.srcnn import SRCNN
from skimage.transform import resize
# import skimage.measure as measure
import skimage.metrics as measure

import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from ztools import *
import yaml
from attrdict import AttrMap
from ndvi_loss.ssim_loss import SSIMLoss
from ndvi_loss.mssim_l1 import MS_SSIM_L1_LOSS
from ndvi_loss.mssim_l1_1 import MS_SSIM_L1_LOSS2


def get_args():
    with open('config/att_damodule_unet1.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    args = AttrMap(config)
    return args


def get_log(args, filen):
    dirname = os.path.join(args.log_dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = dirname + '/' + filen + '.log'
    logging.basicConfig(filename=filename, level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
    return logging


def get_model(args, device):
    m = None
    m_name = 'UNet'
    if args == '0':
        m = UNet(1, 1).to(device)
        m_name = 'UNet1024'
    if args == '1':
        m = UNet6_512(1, 1).to(device)
        m_name = 'UNet6_512'
    if args == '2':
        m = UNet5_256(1, 1).to(device)
        m_name = 'UNet5_256'
    if args == '3':
        m = UNet5_64(1, 1).to(device)
        m_name = 'Unet5_64'
    if args == '4':
        m = UNet4_128(1, 1).to(device)
        m_name = 'Unet4_128'
    if args == '5':
        m = UNet4_32(1, 1).to(device)
        m_name = 'Unet4_32'
    if args == '6':
        m = UNet3_32(1, 1).to(device)
        m_name = 'Unet3_32'
    if args == '7':
        m = Unet6_512_DAUNet1(1, 1).to(device)
        m_name = 'Unet6_512_DAUNet1'
    if args == '8':
        m = Unet6_512_DAUNet2(1, 1).to(device)
        m_name = 'Unet6_512_DAUNet2'
    if args == '9':
        m = Unet6_512_DAUNet3(1, 1).to(device)
        m_name = 'Unet6_512_DAUNet3'
    if args == '71':
        m = Unet6_512_DAUNet4(1, 1).to(device)
        m_name = 'Unet6_512_DAUNet4'
    if args == '81':
        m = Unet6_512_DAUNet5(1, 1).to(device)
        m_name = 'Unet6_512_DAUNet5'
    if args == '91':
        m = Unet6_512_DAUNet6(1, 1).to(device)
        m_name = 'Unet6_512_DAUNet6'
    if args == '72':
        m = Unet6_512_DAUNet7(1, 1).to(device)
        m_name = 'Unet6_512_DAUNet7'
    if args == '82':
        m = Unet6_512_DAUNet8(1, 1).to(device)
        m_name = 'Unet6_512_DAUNet8'
    if args == '92':
        m = Unet6_512_DAUNet9(1, 1).to(device)
        m_name = 'Unet6_512_DAUNet9'
    if args == '93':
        m = Unet6_512_DAUNet10(1, 1).to(device)
        m_name = 'Unet6_512_DAUNet10'
    if args == '94':
        m = Unet6_512_DAUNet11(1, 1).to(device)
        m_name = 'Unet6_512_DAUNet11'
    if args == '95':
        m = Unet6_512_DAUNet12(1, 1).to(device)
        m_name = 'Unet6_512_DAUNet12'

    if args == '16':
        m = Unet6_512_CBAM1(1, 1).to(device)
        m_name = 'Unet6_512_CBAM1'
    if args == '17':
        m = Unet6_512_CBAM2(1, 1).to(device)
        m_name = 'Unet6_512_CBAM2'
    if args == '18':
        m = Unet6_512_CBAM3(1, 1).to(device)
        m_name = 'Unet6_512_CBAM3'
    if args == '19':
        m = Unet6_512_CBAM4(1, 1).to(device)
        m_name = 'Unet6_512_CBAM4'
    if args == '161':
        m = Unet6_512_CBAM5(1, 1).to(device)
        m_name = 'Unet6_512_CBAM5'
    if args == '171':
        m = Unet6_512_CBAM6(1, 1).to(device)
        m_name = 'Unet6_512_CBAM6'
    if args == '181':
        m = Unet6_512_CBAM7(1, 1).to(device)
        m_name = 'Unet6_512_CBAM7'
    if args == '191':
        m = Unet6_512_CBAM8(1, 1).to(device)
        m_name = 'Unet6_512_CBAM8'
    if args == '162':
        m = Unet6_512_CBAM9(1, 1).to(device)
        m_name = 'Unet6_512_CBAM9'
    if args == '172':
        m = Unet6_512_CBAM10(1, 1).to(device)
        m_name = 'Unet6_512_CBAM10'
    if args == '182':
        m = Unet6_512_CBAM11(1, 1).to(device)
        m_name = 'Unet6_512_CBAM11'
    if args == '192':
        m = Unet6_512_CBAM12(1, 1).to(device)
        m_name = 'Unet6_512_CBAM12'

    if args == '26':
        m = Unet6_512_BAM1(1, 1).to(device)
        m_name = 'Unet6_512_BAM1'
    if args == '27':
        m = Unet6_512_BAM2(1, 1).to(device)
        m_name = 'Unet6_512_BAM2'
    if args == '28':
        m = Unet6_512_BAM3(1, 1).to(device)
        m_name = 'Unet6_512_BAM3'
    if args == '29':
        m = Unet6_512_BAM4(1, 1).to(device)
        m_name = 'Unet6_512_BAM4'

    if args == '36':
        m = Unet6_512_CoTAttention1(1, 1).to(device)
        m_name = 'Unet6_512_CoTAttention1'
    if args == '37':
        m = Unet6_512_CoTAttention2(1, 1).to(device)
        m_name = 'Unet6_512_CoTAttention2'
    if args == '38':
        m = Unet6_512_CoTAttention3(1, 1).to(device)
        m_name = 'Unet6_512_CoTAttention3'
    if args == '39':
        m = Unet6_512_CoTAttention4(1, 1).to(device)
        m_name = 'Unet6_512_CoTAttention4'
    if args == '361':
        m = Unet6_512_CoTAttention5(1, 1).to(device)
        m_name = 'Unet6_512_CoTAttention5'
    if args == '371':
        m = Unet6_512_CoTAttention6(1, 1).to(device)
        m_name = 'Unet6_512_CoTAttention6'
    if args == '381':
        m = Unet6_512_CoTAttention7(1, 1).to(device)
        m_name = 'Unet6_512_CoTAttention7'
    if args == '391':
        m = Unet6_512_CoTAttention8(1, 1).to(device)
        m_name = 'Unet6_512_CoTAttention8'
    if args == '362':
        m = Unet6_512_CoTAttention9(1, 1).to(device)
        m_name = 'Unet6_512_CoTAttention9'
    if args == '372':
        m = Unet6_512_CoTAttention10(1, 1).to(device)
        m_name = 'Unet6_512_CoTAttention10'
    if args == '382':
        m = Unet6_512_CoTAttention11(1, 1).to(device)
        m_name = 'Unet6_512_CoTAttention11'
    if args == '392':
        m = Unet6_512_CoTAttention12(1, 1).to(device)
        m_name = 'Unet6_512_CoTAttention12'

    if args == '46':
        m = Unet6_512_PolarizedAttention1(1, 1).to(device)
        m_name = 'Unet6_512_PolarizedAttention1'
    if args == '47':
        m = Unet6_512_PolarizedAttention2(1, 1).to(device)
        m_name = 'Unet6_512_PolarizedAttention2'
    if args == '48':
        m = Unet6_512_PolarizedAttention3(1, 1).to(device)
        m_name = 'Unet6_512_PolarizedAttention3'
    if args == '49':
        m = Unet6_512_PolarizedAttention4(1, 1).to(device)
        m_name = 'Unet6_512_PolarizedAttention4'
    if args == '461':
        m = Unet6_512_PolarizedAttention5(1, 1).to(device)
        m_name = 'Unet6_512_PolarizedAttention5'
    if args == '471':
        m = Unet6_512_PolarizedAttention6(1, 1).to(device)
        m_name = 'Unet6_512_PolarizedAttention6'
    if args == '481':
        m = Unet6_512_PolarizedAttention7(1, 1).to(device)
        m_name = 'Unet6_512_PolarizedAttention7'
    if args == '491':
        m = Unet6_512_PolarizedAttention8(1, 1).to(device)
        m_name = 'Unet6_512_PolarizedAttention8'
    if args == '462':
        m = Unet6_512_PolarizedAttention9(1, 1).to(device)
        m_name = 'Unet6_512_PolarizedAttention9'
    if args == '472':
        m = Unet6_512_PolarizedAttention10(1, 1).to(device)
        m_name = 'Unet6_512_PolarizedAttention10'
    if args == '482':
        m = Unet6_512_PolarizedAttention11(1, 1).to(device)
        m_name = 'Unet6_512_PolarizedAttention11'
    if args == '492':
        m = Unet6_512_PolarizedAttention12(1, 1).to(device)
        m_name = 'Unet6_512_PolarizedAttention12'

    if args == '56':
        m = Unet6_512_PSA1(1, 1).to(device)
        m_name = 'Unet6_512_PSA1'
    if args == '57':
        m = Unet6_512_PSA2(1, 1).to(device)
        m_name = 'Unet6_512_PSA2'
    if args == '58':
        m = Unet6_512_PSA3(1, 1).to(device)
        m_name = 'Unet6_512_PSA3'
    if args == '59':
        m = Unet6_512_PSA4(1, 1).to(device)
        m_name = 'Unet6_512_PSA4'
    if args == '561':
        m = Unet6_512_PSA5(1, 1).to(device)
        m_name = 'Unet6_512_PSA5'
    if args == '571':
        m = Unet6_512_PSA6(1, 1).to(device)
        m_name = 'Unet6_512_PSA6'
    if args == '581':
        m = Unet6_512_PSA7(1, 1).to(device)
        m_name = 'Unet6_512_PSA7'
    if args == '591':
        m = Unet6_512_PSA8(1, 1).to(device)
        m_name = 'Unet6_512_PSA8'
    if args == '562':
        m = Unet6_512_PSA9(1, 1).to(device)
        m_name = 'Unet6_512_PSA9'
    if args == '572':
        m = Unet6_512_PSA10(1, 1).to(device)
        m_name = 'Unet6_512_PSA10'
    if args == '582':
        m = Unet6_512_PSA11(1, 1).to(device)
        m_name = 'Unet6_512_PSA11'
    if args == '592':
        m = Unet6_512_PSA12(1, 1).to(device)
        m_name = 'Unet6_512_PSA12'

    if args == '66':
        m = Unet6_512_SEAttention1(1, 1).to(device)
        m_name = 'Unet6_512_SEAttention1'
    if args == '67':
        m = Unet6_512_SEAttention2(1, 1).to(device)
        m_name = 'Unet6_512_SEAttention2'
    if args == '68':
        m = Unet6_512_SEAttention3(1, 1).to(device)
        m_name = 'Unet6_512_SEAttention3'
    if args == '69':
        m = Unet6_512_SEAttention4(1, 1).to(device)
        m_name = 'Unet6_512_SEAttention4'
    if args == '661':
        m = Unet6_512_SEAttention5(1, 1).to(device)
        m_name = 'Unet6_512_SEAttention5'
    if args == '671':
        m = Unet6_512_SEAttention6(1, 1).to(device)
        m_name = 'Unet6_512_SEAttention6'
    if args == '681':
        m = Unet6_512_SEAttention7(1, 1).to(device)
        m_name = 'Unet6_512_SEAttention7'
    if args == '691':
        m = Unet6_512_SEAttention8(1, 1).to(device)
        m_name = 'Unet6_512_SEAttention8'
    if args == '662':
        m = Unet6_512_SEAttention9(1, 1).to(device)
        m_name = 'Unet6_512_SEAttention9'
    if args == '672':
        m = Unet6_512_SEAttention10(1, 1).to(device)
        m_name = 'Unet6_512_SEAttention10'
    if args == '682':
        m = Unet6_512_SEAttention11(1, 1).to(device)
        m_name = 'Unet6_512_SEAttention11'
    if args == '692':
        m = Unet6_512_SEAttention12(1, 1).to(device)
        m_name = 'Unet6_512_SEAttention12'

    if args == '76':
        m = Unet6_512_ShuffleAttention1(1, 1).to(device)
        m_name = 'Unet6_512_ShuffleAttention1'
    if args == '77':
        m = Unet6_512_ShuffleAttention2(1, 1).to(device)
        m_name = 'Unet6_512_ShuffleAttention2'
    if args == '78':
        m = Unet6_512_ShuffleAttention3(1, 1).to(device)
        m_name = 'Unet6_512_ShuffleAttention3'
    if args == '79':
        m = Unet6_512_ShuffleAttention4(1, 1).to(device)
        m_name = 'Unet6_512_ShuffleAttention4'
    if args == '761':
        m = Unet6_512_ShuffleAttention5(1, 1).to(device)
        m_name = 'Unet6_512_ShuffleAttention5'
    if args == '771':
        m = Unet6_512_ShuffleAttention6(1, 1).to(device)
        m_name = 'Unet6_512_ShuffleAttention6'
    if args == '781':
        m = Unet6_512_ShuffleAttention7(1, 1).to(device)
        m_name = 'Unet6_512_ShuffleAttention7'
    if args == '791':
        m = Unet6_512_ShuffleAttention8(1, 1).to(device)
        m_name = 'Unet6_512_ShuffleAttention8'
    if args == '762':
        m = Unet6_512_ShuffleAttention9(1, 1).to(device)
        m_name = 'Unet6_512_ShuffleAttention9'
    if args == '772':
        m = Unet6_512_ShuffleAttention10(1, 1).to(device)
        m_name = 'Unet6_512_ShuffleAttention10'
    if args == '782':
        m = Unet6_512_ShuffleAttention11(1, 1).to(device)
        m_name = 'Unet6_512_ShuffleAttention11'
    if args == '792':
        m = Unet6_512_ShuffleAttention12(1, 1).to(device)
        m_name = 'Unet6_512_ShuffleAttention12'

    if args == '80':
        m = SRCNN(3)
        m_name = 'SRCNN'
    # if args == '51':
    #     m = VAE(args)
    #     m_name = 'VAE'
    # if args == '52':
    #     m = AttU_Net(1, 1)
    #     m_name = 'AttU_Net_Simple'

    return m, m_name


def get_dataset(args, dataset, device):
    if dataset == '1':
        trains, vals, tests = pkl.load(open('data/unet_trainset1.pkl', 'rb'))
    if dataset == '2':
        trains, vals, tests = pkl.load(open('data/unet_trainset5.pkl', 'rb'))
    if dataset == '3':
        trains, vals, tests = pkl.load(open('data/unet_trainset3.pkl', 'rb'))
    if dataset == '4':
        trains, vals, tests = pkl.load(open('data/unet_trainset4.pkl', 'rb'))
    tr_data = NdviDataset(args.data_dir, trains, False, device, transform=x_transforms, target_transform=y_transforms)
    tr_loader = DataLoader(tr_data, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
    val_data = NdviDataset(args.data_dir, vals, False, device, transform=x_transforms, target_transform=y_transforms)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = val_loader
    return tr_loader, val_loader, test_loader


def train(model, train_loader, criterion, epoch, args):
    model.train()
    num_batches = 0
    avg_loss = []
    avg_loss1 = []
    dt_size = len(train_loader.dataset)
    steps = 0
    # ---------------------------- #
    # gradient accumulate 1
    # ---------------------------- #
    for x, y, _, mask in train_loader:
        # print(len(x))
        inputs = x.to(device)
        labels = y.to(device)
        output = model(inputs)
        loss = criterion(output, labels)
        avg_loss.append(loss.item())
        loss = loss / args.accumulation_steps  # 损失标准化
        loss.backward()
        num_batches += 1
        if num_batches % args.accumulation_steps == 0:
            steps += 1
            avg_loss1.append(loss.item())
            optimizer.step()  # 更新参数
            optimizer.zero_grad()  # 梯度清零

        if num_batches % args.log_step == 0:
            print("%d/%d,loss:%0.6f" % (num_batches, (dt_size - 1) // train_loader.batch_size + 1, loss.item()))
            logging.info("%d/%d,loss:%0.6f" % (num_batches, (dt_size - 1) // train_loader.batch_size + 1, loss.item()))

    # avg_loss /= num_batches
    # avg_loss1 /= steps
    print('epoch: {} train_loss1:{} train_loss2:{}'.format(epoch, np.mean(avg_loss), np.mean(avg_loss1)))
    logging.info('epoch: {} train_loss1:{} train_loss2:{}'.format(epoch, np.mean(avg_loss), np.mean(avg_loss1)))


def get_metrics(image_mask, predict):
    # mse = measure.compare_mse(image_mask, predict)
    # psnr = measure.compare_psnr(image_mask, predict, data_range=2)
    # ssim = measure.compare_ssim(image_mask, predict)
    mse = measure.mean_squared_error(image_mask, predict)
    psnr = measure.peak_signal_noise_ratio(image_mask, predict, data_range=2)
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
                inputs = x.to(device)
                labels = y.to(device)
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
            save_path = os.path.join(args.checkpoints, file_name + ".pt")
            if val_loss < best_loss:
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
        data, img_id, height, width = sample_batched['image'], sample_batched['img_id'], sample_batched['height'], \
                                      sample_batched['width']
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

    # print('cuda id')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # str(input())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = get_args()

    print('model name')
    model_order = '80'  # input()

    print('loss function(mse/ssim/mssim_l1/mssim_l11)')
    args.rec_loss = "mse"  # input()

    print('additional information')
    model_alias = 'lr001'  # input()

    print('batch size')
    args.batch_size = 16  # int(input())

    # print('scheduler name(step/expo)')
    scheduler_name = 'expo'  # input()

    # print('dataset number(1/2/3/4)')
    # dataset_number = input()
    dataset_number = '2'

    print('learning rate')
    args.learning_rate = 0.001  # float(input())

    print('weight decay')
    args.weight_decay = 0.0001  # float(input())
    model, model_name = get_model(model_order, device)
    args.arch = model_name

    file_name = model_name + '_' + args.rec_loss + '_data' + str(dataset_number) + '_e' + str(args.epoch) + '_b' + str(
        args.batch_size) + '_' + model_alias
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

    train_loader, val_loader, test_loader = get_dataset(args, dataset_number, device)
    # starting params
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
