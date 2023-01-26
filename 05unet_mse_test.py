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
from ndvi_models.unet_danet import Unet6_512_DAUNet7, Unet6_512_DAUNet8, Unet6_512_DAUNet9, Unet6_512_DAUNet10, Unet6_512_DAUNet11, Unet6_512_DAUNet12
from ndvi_models.unet_cbam import Unet6_512_CBAM1, Unet6_512_CBAM2, Unet6_512_CBAM3, Unet6_512_CBAM4
from ndvi_models.unet_cbam import Unet6_512_CBAM5, Unet6_512_CBAM6, Unet6_512_CBAM7, Unet6_512_CBAM8
from ndvi_models.unet_cbam import Unet6_512_CBAM9, Unet6_512_CBAM10, Unet6_512_CBAM11, Unet6_512_CBAM12
from ndvi_models.unet_bam import Unet6_512_BAM1, Unet6_512_BAM2, Unet6_512_BAM3, Unet6_512_BAM4
from ndvi_models.unet_CoTAttention import Unet6_512_CoTAttention1, Unet6_512_CoTAttention2, Unet6_512_CoTAttention3, Unet6_512_CoTAttention4
from ndvi_models.unet_CoTAttention import Unet6_512_CoTAttention5, Unet6_512_CoTAttention6, Unet6_512_CoTAttention7, Unet6_512_CoTAttention8
from ndvi_models.unet_CoTAttention import Unet6_512_CoTAttention9, Unet6_512_CoTAttention10, Unet6_512_CoTAttention11, Unet6_512_CoTAttention12
from ndvi_models.unet_polarizedselfattention import Unet6_512_PolarizedAttention1, Unet6_512_PolarizedAttention2, Unet6_512_PolarizedAttention3, Unet6_512_PolarizedAttention4
from ndvi_models.unet_polarizedselfattention import Unet6_512_PolarizedAttention5, Unet6_512_PolarizedAttention6, Unet6_512_PolarizedAttention7, Unet6_512_PolarizedAttention8
from ndvi_models.unet_polarizedselfattention import Unet6_512_PolarizedAttention9, Unet6_512_PolarizedAttention10, Unet6_512_PolarizedAttention11, Unet6_512_PolarizedAttention12
from ndvi_models.unet_psa import Unet6_512_PSA1, Unet6_512_PSA2, Unet6_512_PSA3, Unet6_512_PSA4
from ndvi_models.unet_psa import Unet6_512_PSA5, Unet6_512_PSA6, Unet6_512_PSA7, Unet6_512_PSA8
from ndvi_models.unet_psa import Unet6_512_PSA9, Unet6_512_PSA10, Unet6_512_PSA11, Unet6_512_PSA12
from ndvi_models.unet_seattention import Unet6_512_SEAttention1, Unet6_512_SEAttention2, Unet6_512_SEAttention3, Unet6_512_SEAttention4
from ndvi_models.unet_seattention import Unet6_512_SEAttention5, Unet6_512_SEAttention6, Unet6_512_SEAttention7, Unet6_512_SEAttention8
from ndvi_models.unet_seattention import Unet6_512_SEAttention9, Unet6_512_SEAttention10, Unet6_512_SEAttention11, Unet6_512_SEAttention12
from ndvi_models.unet_shuffleattention import Unet6_512_ShuffleAttention1, Unet6_512_ShuffleAttention2, Unet6_512_ShuffleAttention3, Unet6_512_ShuffleAttention4
from ndvi_models.unet_shuffleattention import Unet6_512_ShuffleAttention5, Unet6_512_ShuffleAttention6, Unet6_512_ShuffleAttention7, Unet6_512_ShuffleAttention8
from ndvi_models.unet_shuffleattention import Unet6_512_ShuffleAttention9, Unet6_512_ShuffleAttention10, Unet6_512_ShuffleAttention11, Unet6_512_ShuffleAttention12
from skimage.transform import resize
import skimage.measure as measure
# import skimage.metrics as measure

import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from ztools import *
import yaml
from attrdict import AttrMap
from ndvi_loss.ssim_loss import SSIMLoss
from ndvi_loss.mssim_l1 import MS_SSIM_L1_LOSS
from ndvi_loss.mssim_l1_1 import MS_SSIM_L1_LOSS2


def get_args():
    with open('config/att_unet1.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    args = AttrMap(config)
    return args


def get_log(args, filen):
    dirname = os.path.join(args.log_dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = dirname + '/' + filen + 'test.log'
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
    # if args == '50':
    #     m = RegressionCNN()
    #     m_name = 'CNNReg'
    # if args == '51':
    #     m = VAE(args)
    #     m_name = 'VAE'
    # if args == '52':
    #     m = AttU_Net(1, 1)
    #     m_name = 'AttU_Net_Simple'

    return m, m_name


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

    te_data = NdviDataset(args.data_dir, tests, False, transform=x_transforms, target_transform=y_transforms)
    test_loader = DataLoader(te_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return tr_loader, val_loader, test_loader


def get_metrics(image_mask, predict):
    mse = measure.compare_mse(image_mask, predict)
    psnr = measure.compare_psnr(image_mask, predict, data_range=2)
    # ssim = measure.compare_ssim(image_mask, predict)
    # mse = measure.mean_squared_error(image_mask, predict)
    # psnr = measure.peak_signal_noise_ratio(image_mask, predict, data_range=2)
    # ssim = measure.structural_similarity(image_mask, predict, data_range=2,multichannel=True)
    rmse = math.sqrt(mse)
    return rmse, psnr  # , ssim


def run_test(model, test_loader, criterion):
    model.eval()
    rmse_sum = []
    psnr_sum = []
    ssim_sum = []
    tests = pkl.load(open('data/unet_test_random_points.pkl', 'rb'))
    source_pair = []
    pred_pair = []
    with torch.no_grad():
        for x, y, name, mask in test_loader:
            print(name)
            if x.shape[0] > 0:
                inputs = x.to(device)
                labels = y.to(device)
                output = model(inputs)

                rmse, psnr = get_metrics(labels.detach().cpu().numpy(), output.detach().cpu().numpy())
                rmse_sum.append(rmse)
                psnr_sum.append(psnr)
                # ssim_sum.append(ssim)
                input1 = inputs.data.cpu().numpy()
                output = output.data.cpu().numpy()
                labels = labels.detach().cpu().numpy()
                for i in range(0, output.shape[0]):
                    nam = name[i].split('/')
                    nam = nam[-1].split('.')[0]
                    inp = np.squeeze(input1[i])
                    pred = np.squeeze(output[i])
                    label = np.squeeze(labels[i])
                    save_test(inp, pred, label, nam)

                    pts = tests[int(nam)]
                    for pt in pts:
                        pred_pair.append([label[pt[0], pt[1]], pred[pt[0], pt[1]]])
                        source_pair.append([label[pt[0], pt[1]], inp[pt[0], pt[1]]])

                pkl.dump(pred_pair, open('result/test/unet_test_pred_pair.pkl', 'wb'))
                pkl.dump(source_pair, open('result/test/unet_test_source_pair.pkl', 'wb'))

    print('test: ' + ', rmse: ' + str(np.mean(rmse_sum)))
    logging.info('test: ' + ', rmse: ' + str(np.mean(rmse_sum)))
    logging.info('test: ' + ', psnr: ' + str(np.mean(psnr_sum)))


def save_test(inp, pred, label, i):
    outdir = '/home/hb/work/tmp/unetout'
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 3, 1)
    plt.imshow(inp, cmap='PuBuGn')
    ax3 = fig.add_subplot(1, 3, 2)
    plt.imshow(pred, cmap='PuBuGn')
    ax3 = fig.add_subplot(1, 3, 3)
    plt.imshow(label, cmap='PuBuGn')
    plt.savefig(os.path.join(outdir, "{}.jpg".format(i)))
    plt.close()


if __name__ == '__main__':
    x_transforms = transforms.ToTensor()
    y_transforms = transforms.ToTensor()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_args()
    with open('config/test_list.txt', 'r') as f:
        lines = f.readlines()

    for l in lines:
        ll = l.split(' ')
        model_order = ll[0]
        args.batch_size = 1
        dataset_number = '2'
        model, model_name = get_model(model_order, device)
        args.arch = model_name

        file_name = 'all_test'
        logging = get_log(args, file_name)
        train_loader, val_loader, test_loader = get_dataset(args, dataset_number)
        checkpoint = torch.load(args.test_model)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.cuda()
        run_test(model, test_loader, args)
