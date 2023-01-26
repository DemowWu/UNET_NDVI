import matplotlib.pyplot as plt
import pickle as pkl
import os

logfile = '/media/hb/data/pywork/paper/ndvi2/UNet_NDVI/result/log2/Unet31_mse_sch_expo_data2_ep200_bat16_oneconv.log'


def plot_total_loss():
    with open(logfile, 'r') as f1:
        list1 = f1.readlines()
        train_loss1 = []
        train_loss2 = []
        val_loss = []
        for l in list1:
            if 'epoch' in l and 'val_loss:' in l:
                s1 = l.split('val_loss:')
                val_loss.append(float(s1[1]))
            if 'epoch' in l and 'train_loss1:' in l:
                s1 = l.split('train_loss1:')
                s2 = s1[1].split('train_loss2:')
                train_loss1.append(float(s2[0].strip()))
                train_loss2.append(float(s2[1].strip()))
                # s1 = l.split('train_loss:')
                # train_loss.append(float(s1[1]))

        x = [i for i in range(len(train_loss1))]
        # fig = plt.figure(figsize=(10, 15))
        # ax1 = fig.add_subplot(2, 1, 1)
        # plt.plot(x, losses, marker='o', mec='r', mfc='w', label='train losses')
        # plt.legend()
        # plt.xlabel("epochs")train_loss2
        # plt.ylabel("train losses")
        # ax2 = fig.add_subplot(2, 1, 2)
        # plt.plot(x, vallosses, marker='o', mec='r', mfc='w', label='val losses')
        # plt.legend()
        # plt.xlabel("epochs")
        # plt.ylabel("val losses")
        # plt.show()

        plt.plot(x, train_loss1, marker='*', mec='r', mfc='w', label='train_loss1')
        # plt.plot(x, train_loss2, marker='+', mec='b', mfc='w', label='train_loss2')
        plt.plot(x, val_loss, marker='o', mec='g', mfc='w', label='val_loss')
        # # plt.plot(x, y1, marker='*', ms=10, label=u'y=x^3曲线图')
        plt.legend()  # 让图例生效
        # plt.margins(0)
        # plt.subplots_adjust(bottom=0.15)
        # plt.xlabel("epochs")  # X轴标签
        # plt.ylabel("total loss")  # Y轴标签
        # # plt.title("A simple plot")  # 标题
        plt.show()


def get_loss(logfile):
    with open(logfile, 'r') as f1:
        list1 = f1.readlines()
        train_loss1 = []
        train_loss2 = []
        val_loss = []
        val_rmse = []
        val_psnr = []
        for l in list1:
            if 'epoch' in l and 'val_loss:' in l:
                s1 = l.split('val_loss:')
                val_loss.append(float(s1[1]))
            if 'epoch' in l and 'train_loss1:' in l:
                s1 = l.split('train_loss1:')
                s2 = s1[1].split('train_loss2:')
                train_loss1.append(float(s2[0].strip()))
                train_loss2.append(float(s2[1].strip()))
            if 'epoch' in l and 'rmse:' in l:
                s1 = l.split('rmse:')
                val_rmse.append(float(s1[1]))
            if 'epoch' in l and 'psnr:' in l:
                s1 = l.split('psnr:')
                val_psnr.append(float(s1[1]))
    return train_loss1, val_loss, val_rmse, val_psnr


def plot_multi_train(filenames, smooth_factor=0.9):
    ndir = '/media/hb/data/pywork/paper/ndvi2/UNet_NDVI/result/log2'
    fig = plt.figure(figsize=(15, 9))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    ax1 = fig.add_subplot(2, 2, 1)
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=1, wspace=None, hspace=None)
    # train loss
    plt.title('train loss')
    for k in filenames.keys():
        train_loss1, val_loss, val_rmse, val_psnr = get_loss(os.path.join(ndir, filenames[k]))
        train_loss1 = smooth_curve(train_loss1, smooth_factor)
        x = [i for i in range(len(train_loss1))]
        plt.plot(x, train_loss1, label=k)
        plt.legend()

    ax1 = fig.add_subplot(2, 2, 2)
    plt.title('validation loss')
    for k in filenames.keys():
        train_loss1, val_loss, val_rmse, val_psnr = get_loss(os.path.join(ndir, filenames[k]))
        val_loss = smooth_curve(val_loss, smooth_factor)
        x = [i for i in range(len(val_loss))]
        plt.plot(x, val_loss, label=k)
        plt.legend()

    ax1 = fig.add_subplot(2, 2, 4)
    plt.title('validation rmse')
    for k in filenames.keys():
        train_loss1, val_loss, val_rmse, val_psnr = get_loss(os.path.join(ndir, filenames[k]))
        val_rmse = smooth_curve(val_rmse, smooth_factor)
        x = [i for i in range(len(val_rmse))]
        plt.plot(x, val_rmse, label=k, linewidth=2)
        plt.legend()

    ax1 = fig.add_subplot(2, 2, 3)
    plt.title('validation psnr')
    for k in filenames.keys():
        train_loss1, val_loss, val_rmse, val_psnr = get_loss(os.path.join(ndir, filenames[k]))
        val_psnr = smooth_curve(val_psnr, smooth_factor)
        x = [i for i in range(len(val_psnr))]
        plt.plot(x, val_psnr, label=k, linewidth=2)
        plt.legend()

    fig.set_tight_layout(True)
    plt.show()


def plot_multi_rmse(filenames):
    ndir = '/home/hb/work/pywork/ndvi2/UNet_NDVI/result/log'
    fig = plt.figure(figsize=(12, 9))
    for k in filenames.keys():
        train_loss1, val_loss, val_rmse, val_psnr = get_loss(os.path.join(ndir, filenames[k]))
        x = [i for i in range(len(val_rmse))]
        plt.plot(x, val_rmse, label=k)
        plt.legend()
    plt.show()


def plot_multi_psnr(filenames):
    fig = plt.figure(figsize=(12, 9))
    ndir = '/home/hb/work/pywork/ndvi2/UNet_NDVI/result/log'
    for k in filenames.keys():
        train_loss1, val_loss, val_rmse, val_psnr = get_loss(os.path.join(ndir, filenames[k]))
        x = [i for i in range(len(val_psnr))]
        plt.plot(x, val_psnr, label=k, linewidth=4)
        plt.legend()
    plt.show()


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)

    return smoothed_points


def plot_multi_psnr_rmse(filenames, smooth_factor=0.9):
    fig = plt.figure(figsize=(18.5, 8))
    ndir = '/home/hb/pywork/paper/ndvi2/UNet_NDVI/result/log2'
    ax1 = fig.add_subplot(1, 2, 1)
    plt.title('validation psnr')
    for k in filenames.keys():
        train_loss1, val_loss, val_rmse, val_psnr = get_loss(os.path.join(ndir, filenames[k]))
        x = [i for i in range(len(val_psnr))]
        val_psnr = smooth_curve(val_psnr, smooth_factor)
        plt.plot(x, val_psnr, label=k, linewidth=2)
        plt.legend()
        print('psnr',k,val_psnr[-1])

    ax1 = fig.add_subplot(1, 2, 2)
    plt.title('validation rmse')
    for k in filenames.keys():
        train_loss1, val_loss, val_rmse, val_psnr = get_loss(os.path.join(ndir, filenames[k]))
        x = [i for i in range(len(val_rmse))]
        val_rmse = smooth_curve(val_rmse, smooth_factor)
        plt.plot(x, val_rmse, label=k, linewidth=2)
        plt.legend()
        print('rmse',k,val_rmse[-1])

    fig.set_tight_layout(True)
    plt.show()


unet5_256_logs = {
    # 'Unet3_32_twoconv_01_decay001': 'Unet3_32_mse_sch_expo_data2_ep200_bat16_01.log',
    # 'Unet3_32_twoconv_001aug': 'Unet3_32_mse_sch_expo_data2_ep200_bat16_001aug.log',
    # 'Unet3_32_twoconv_aug01':'Unet3_32_mse_sch_expo_data2_ep200_bat16_aug01.log',
    # 'Unet3_32_twoconv_01_decay00001': 'Unet3_32_mse_sch_expo_data2_ep200_bat16_decay00001.log',
    # 'Unet3_32_twoconv_01':'Unet3_32_mse_sch_expo_data2_ep200_bat16_lr01.log',
    # 'Unet3_32_twoconv_001_decay001': 'Unet3_32_mse_sch_expo_data2_ep200_bat16_lr001decay001.log',
    # 'Unet3_32_oneconv_01': 'Unet3_32_mse_sch_expo_data2_ep200_bat16_oneconv01.log',
    # 'Unet3_32_oneconv_001': 'Unet3_32_mse_sch_expo_data2_ep200_bat16_onecov001.log',
    # 'Unet3_32_oneconv_0001': 'Unet3_32_mse_sch_expo_data2_ep200_bat16_onconv0001.log',
    # 'Unet4_128_001': 'Unet4_128_mse_data2_e200_b16_lr001.log',
    # 'Unet5_64': 'Unet5_64_mse_data2_e200_b16_lr001.log',

    # first round result
    # 'Unet3_32_twoconv_001': 'Unet3_32_mse_sch_expo_data2_ep200_bat16_lr001.log',
    # 'Unet3_32_twoconv_001_decay00001': 'Unet3_32_mse_sch_expo_data2_ep200_bat16_lr001decay00001.log',
    # 'Unet4_32_001': 'Unet4_32_mse_data2_e200_b16_lr001.log',   # *
    # 'Unet5_256': 'Unet5_256_mse_data2_e200_b16_lr001.log',     # *
    'Unet6_512': 'Unet6_512_mse_data2_e200_b16_lr001.log',  # **

    # second round result
    # 'Unet6_512_DAUNet1_mse': 'Unet6_512_DAUNet1_mse_data2_e200_b16_lr001.log',
    # 'Unet6_512_DAUNet1_ssmi': 'Unet6_512_DAUNet1_ssim_data2_e200_b16_lr001.log',
    # 'Unet6_512_DAUNet2_mse': 'Unet6_512_DAUNet2_mse_data2_e200_b16_lr001.log',  # **
    # 'Unet6_512_DAUNet2_ssim': 'Unet6_512_DAUNet2_ssim_data2_e200_b16_lr001.log',
    # 'Unet6_512_DAUNet2_ssim_l1': 'Unet6_512_DAUNet2_mssim_l11_data2_e200_b16_lr001.log',
    # 'Unet6_512_DAUNet1_mse8': 'Unet6_512_DAUNet1_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_DAUNet2_mse8': 'Unet6_512_DAUNet2_mse_data2_e200_b8_lr001.log',
    'Unet6_512_DAUNet3_mse8': 'Unet6_512_DAUNet3_mse_data2_e200_b8_lr001.log', # 这个好
    # 'Unet6_512_DAUNet4_mse8': 'Unet6_512_DAUNet4_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_DAUNet5_mse8': 'Unet6_512_DAUNet5_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_DAUNet6_mse8': 'Unet6_512_DAUNet6_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_DAUNet7_mse8': 'Unet6_512_DAUNet7_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_DAUNet8_mse8': 'Unet6_512_DAUNet8_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_DAUNet9_mse8': 'Unet6_512_DAUNet9_mse_data2_e200_b16_lr001.log',
    # 'Unet6_512_DAUNet10_mse8': 'Unet6_512_DAUNet10_mse_data2_e200_b16_lr001.log',
    # 'Unet6_512_DAUNet11_mse8': 'Unet6_512_DAUNet11_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_DAUNet12_mse8': 'Unet6_512_DAUNet12_mse_data2_e200_b12_lr001.log',

    # 'Unet6_512_CBAM1_mse': 'Unet6_512_CBAM1_mse_data2_e200_b16_lr001.log',
    # 'Unet6_512_CBAM1_ssim': 'Unet6_512_CBAM1_ssim_data2_e200_b16_lr001.log',
    # 'Unet6_512_CBAM2_ssim': 'Unet6_512_CBAM2_ssim_data2_e200_b16_lr001.log',  # **
    # 'Unet6_512_CBAM2_mssim':'Unet6_512_CBAM2_mssim_l11_data2_e200_b24_lr001.log',
    # 'Unet6_512_CBAM3_mse': 'Unet6_512_CBAM3_mse_data2_e200_b16_lr001.log',
    # 'Unet6_512_CBAM3_mse_24':'Unet6_512_CBAM3_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_CBAM3_mssim':'Unet6_512_CBAM3_mssim_l11_data2_e200_b24_lr001.log', #??
    # 'Unet6_512_CBAM4_mse': 'Unet6_512_CBAM4_mse_data2_e200_b16_lr001.log',
    # 'Unet6_512_CBAM5_mse':'Unet6_512_CBAM5_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_CBAM7_mse':'Unet6_512_CBAM7_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_CBAM8_mse':'Unet6_512_CBAM8_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_CBAM9_mse': 'Unet6_512_CBAM9_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_CBAM9_mse_add':'Unet6_512_CBAM9_mse_data2_e200_b24_add_lr001.log',
    # 'Unet6_512_CBAM11_mse': 'Unet6_512_CBAM11_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_CBAM12_mse': 'Unet6_512_CBAM12_mse_data2_e200_b24_lr001.log',

    # 'Unet6_512_CBAM2_mse_8': 'Unet6_512_CBAM2_mse_data2_e200_b8_lr001.log',
    'Unet6_512_CBAM3_mse_8': 'Unet6_512_CBAM3_mse_data2_e200_b8_lr001.log',  # 这个
    # 'Unet6_512_CBAM4_mse_8': 'Unet6_512_CBAM4_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_CBAM5_mse8': 'Unet6_512_CBAM5_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_CBAM6_mse8': 'Unet6_512_CBAM6_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_CBAM7_mse8':'Unet6_512_CBAM7_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_CBAM9_mse8': 'Unet6_512_CBAM9_mse_data2_e200_b8_lr001.log',
    'Unet6_512_CBAM10_mse8': 'Unet6_512_CBAM10_mse_data2_e200_b8_lr001.log', #这个
    # 'Unet6_512_CBAM11_mse8': 'Unet6_512_CBAM11_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_CBAM12_mse8': 'Unet6_512_CBAM12_mse_data2_e200_b8_lr001.log',

    # 'Unet6_512_CBAM2_mse': 'Unet6_512_CBAM2_mse_data2_e200_b16_lr001.log',  # **
    # 'Unet6_512_CBAM6_mse': 'Unet6_512_CBAM6_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_CBAM10_mse': 'Unet6_512_CBAM10_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_CBAM10_mse_add': 'Unet6_512_CBAM10_mse_data2_e200_b24_add_lr001.log',

    #
    # 'Unet6_512_BAM1_mse': 'Unet6_512_BAM1_mse_data2_e200_b16_lr001.log',
    # 'Unet6_512_BAM2_mse': 'Unet6_512_BAM2_mse_data2_e200_b16_lr001.log',          # good
    # 'Unet6_512_BAM3_mse': 'Unet6_512_BAM3_mse_data2_e200_b16_lr001.log',  # **
    # 'Unet6_512_BAM1_mse8': 'Unet6_512_BAM1_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_BAM2_mse8': 'Unet6_512_BAM2_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_BAM3_mse8': 'Unet6_512_BAM3_mse_data2_e200_b8_lr001.log',

    #
    # 'Unet6_512_PolarizedAttention1_mse': 'Unet6_512_PolarizedAttention1_mse_data2_e200_b16_lr001.log',
    # 'Unet6_512_PolarizedAttention2_mse': 'Unet6_512_PolarizedAttention2_mse_data2_e200_b16_lr001.log',
    # 'Unet6_512_PolarizedAttention3_mse': 'Unet6_512_PolarizedAttention3_mse_data2_e200_b16_lr001.log',
    # 'Unet6_512_PolarizedAttention4_mse': 'Unet6_512_PolarizedAttention4_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_PolarizedAttention5_mse': 'Unet6_512_PolarizedAttention5_mse_data2_e200_b32_lr001.log',
    # 'Unet6_512_PolarizedAttention6_mse': 'Unet6_512_PolarizedAttention6_mse_data2_e200_b16_lr001.log',
    # 'Unet6_512_PolarizedAttention7_mse': 'Unet6_512_PolarizedAttention7_mse_data2_e200_b16_lr001.log',
    # 'Unet6_512_PolarizedAttention8_mse': 'Unet6_512_PolarizedAttention8_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_PolarizedAttention9_mse': 'Unet6_512_PolarizedAttention9_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_PolarizedAttention10_add_mse': 'Unet6_512_PolarizedAttention10_mse_data2_e200_b16_add_lr001.log',
    # 'Unet6_512_PolarizedAttention11_add_mse': 'Unet6_512_PolarizedAttention11_mse_data2_e200_b16_add_lr001.log',
    # 'Unet6_512_PolarizedAttention10_mse': 'Unet6_512_PolarizedAttention10_mse_data2_e200_b16_lr001.log',
    # 'Unet6_512_PolarizedAttention10_add_right_mse':'Unet6_512_PolarizedAttention10_mse_data2_e200_b16_add_right_lr001.log',
    # 'Unet6_512_PolarizedAttention11_add_right_mse': 'Unet6_512_PolarizedAttention11_mse_data2_e200_b16_add_right_lr001.log',
    # 'Unet6_512_PolarizedAttention12_mse': 'Unet6_512_PolarizedAttention12_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_PolarizedAttention1_mse8': 'Unet6_512_PolarizedAttention1_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_PolarizedAttention2_mse8': 'Unet6_512_PolarizedAttention2_mse_data2_e200_b8_lr001.log',
    'Unet6_512_PolarizedAttention3_mse8': 'Unet6_512_PolarizedAttention3_mse_data2_e200_b8_lr001.log', # 这个
    # 'Unet6_512_PolarizedAttention4_mse8': 'Unet6_512_PolarizedAttention4_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_PolarizedAttention5_mse8': 'Unet6_512_PolarizedAttention5_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_PolarizedAttention6_mse8': 'Unet6_512_PolarizedAttention6_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_PolarizedAttention7_mse8': 'Unet6_512_PolarizedAttention7_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_PolarizedAttention8_mse8': 'Unet6_512_PolarizedAttention8_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_PolarizedAttention9_mse8': 'Unet6_512_PolarizedAttention9_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_PolarizedAttention10_mse8': 'Unet6_512_PolarizedAttention10_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_PolarizedAttention11_mse8': 'Unet6_512_PolarizedAttention11_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_PolarizedAttention12_mse8': 'Unet6_512_PolarizedAttention12_mse_data2_e200_b8_lr001.log',

    # 'Unet6_512_PSA1_mse': 'Unet6_512_PSA1_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_PSA1_mse8':'Unet6_512_PSA1_mse_data2_e200_b8_lr001.log',
    'Unet6_512_PSA2_mse8': 'Unet6_512_PSA2_mse_data2_e200_b8_lr001.log', # 这个和9 差不多
    # 'Unet6_512_PSA3_mse8': 'Unet6_512_PSA3_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_PSA4_mse8': 'Unet6_512_PSA4_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_PSA5_mse8': 'Unet6_512_PSA5_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_PSA6_mse8': 'Unet6_512_PSA6_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_PSA7_mse8': 'Unet6_512_PSA7_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_PSA8_mse8': 'Unet6_512_PSA8_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_PSA9_mse8': 'Unet6_512_PSA9_mse_data2_e200_b8_lr001.log', # 这个
    # 'Unet6_512_PSA10_mse8': 'Unet6_512_PSA10_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_PSA11_mse8': 'Unet6_512_PSA11_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_PSA12_mse8': 'Unet6_512_PSA12_mse_data2_e200_b8_lr001.log',

    # 'Unet6_512_SEAttention1_mse_24': 'Unet6_512_SEAttention1_mse_data2_e200_b24_lr001_2.log', #很差
    # 'Unet6_512_SEAttention2_mse_24': 'Unet6_512_SEAttention2_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_SEAttention4_mse': 'Unet6_512_SEAttention4_mse_data2_e200_b16_lr001.log',
    # 'Unet6_512_SEAttention5_mse': 'Unet6_512_SEAttention5_mse_data2_e200_b16_lr001.log',
    # 'Unet6_512_SEAttention7_mse_24': 'Unet6_512_SEAttention7_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_SEAttention6_mse_24': 'Unet6_512_SEAttention6_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_SEAttention8_mse':'Unet6_512_SEAttention8_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_SEAttention9_mse':'Unet6_512_SEAttention9_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_SEAttention10_mse_add': 'Unet6_512_SEAttention10_mse_data2_e200_b24_lr001_2.log',
    # 'Unet6_512_SEAttention11_mse_add': 'Unet6_512_SEAttention11_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_SEAttention3_mse_24': 'Unet6_512_SEAttention3_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_SEAttention12': 'Unet6_512_SEAttention12_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_SEAttention1_mse8': 'Unet6_512_SEAttention1_mse_data2_e200_b8_lr001.log',
    'Unet6_512_SEAttention2_mse8': 'Unet6_512_SEAttention2_mse_data2_e200_b8_lr001.log', # 这个
    # 'Unet6_512_SEAttention3_mse8': 'Unet6_512_SEAttention3_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_SEAttention4_mse8': 'Unet6_512_SEAttention4_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_SEAttention5_mse8': 'Unet6_512_SEAttention5_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_SEAttention6_mse8': 'Unet6_512_SEAttention6_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_SEAttention7_mse8': 'Unet6_512_SEAttention7_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_SEAttention8_mse8': 'Unet6_512_SEAttention8_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_SEAttention9_mse8': 'Unet6_512_SEAttention9_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_SEAttention10_mse8': 'Unet6_512_SEAttention10_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_SEAttention11_mse8': 'Unet6_512_SEAttention11_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_SEAttention12_mse8': 'Unet6_512_SEAttention12_mse_data2_e200_b8_lr001.log',

    # 'Unet6_512_CoTAttention1_mse': 'Unet6_512_CoTAttention1_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_CoTAttention1_mse16':'Unet6_512_CoTAttention1_mse_data2_e200_b16_lr001.log',
    # 'Unet6_512_CoTAttention2':'Unet6_512_CoTAttention2_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_CoTAttention1_mse8': 'Unet6_512_CoTAttention1_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_CoTAttention2_mse8': 'Unet6_512_CoTAttention2_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_CoTAttention3_mse8': 'Unet6_512_CoTAttention3_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_CoTAttention4_mse8': 'Unet6_512_CoTAttention4_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_CoTAttention5_mse8': 'Unet6_512_CoTAttention5_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_CoTAttention6_mse8': 'Unet6_512_CoTAttention6_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_CoTAttention7_mse8': 'Unet6_512_CoTAttention7_mse_data2_e200_b8_lr001.log',
    'Unet6_512_CoTAttention7_mse82': 'Unet6_512_CoTAttention7_mse_data2_e200_b8_lr002.log', # 这个
    # 'Unet6_512_CoTAttention7_mse84':'Unet6_512_CoTAttention7_mse_data2_e200_b8_lr004.log',
    # 'Unet6_512_CoTAttention8_mse8': 'Unet6_512_CoTAttention8_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_CoTAttention9_mse8': 'Unet6_512_CoTAttention9_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_CoTAttention10_mse8': 'Unet6_512_CoTAttention10_mse_data2_e200_b8_lr001.log',

    # 'Unet6_512_ShuffleAttention1_mse': 'Unet6_512_ShuffleAttention1_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_ShuffleAttention2_mse':'Unet6_512_ShuffleAttention2_mse_data2_e200_b16_lr001.log',
    # 'Unet6_512_ShuffleAttention3_mse':'Unet6_512_ShuffleAttention3_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_ShuffleAttention4_mse': 'Unet6_512_ShuffleAttention4_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_ShuffleAttention5_mse': 'Unet6_512_ShuffleAttention5_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_ShuffleAttention6_mse': 'Unet6_512_ShuffleAttention6_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_ShuffleAttention7_mse': 'Unet6_512_ShuffleAttention7_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_ShuffleAttention8_mse': 'Unet6_512_ShuffleAttention8_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_ShuffleAttention9_mse': 'Unet6_512_ShuffleAttention9_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_ShuffleAttention10_mse': 'Unet6_512_ShuffleAttention10_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_ShuffleAttention11_mse': 'Unet6_512_ShuffleAttention11_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_ShuffleAttention12_mse': 'Unet6_512_ShuffleAttention12_mse_data2_e200_b24_lr001.log',
    # 'Unet6_512_ShuffleAttention1_mse8':'Unet6_512_ShuffleAttention1_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_ShuffleAttention2_mse8':'Unet6_512_ShuffleAttention2_mse_data2_e200_b8_lr001.log',
    'Unet6_512_ShuffleAttention3_mse8':'Unet6_512_ShuffleAttention3_mse_data2_e200_b8_lr001.log', # 这个
    # 'Unet6_512_ShuffleAttention4_mse8': 'Unet6_512_ShuffleAttention4_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_ShuffleAttention5_mse8': 'Unet6_512_ShuffleAttention5_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_ShuffleAttention6_mse8': 'Unet6_512_ShuffleAttention6_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_ShuffleAttention7_mse8': 'Unet6_512_ShuffleAttention7_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_ShuffleAttention8_mse8': 'Unet6_512_ShuffleAttention8_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_ShuffleAttention9_mse8': 'Unet6_512_ShuffleAttention9_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_ShuffleAttention10_mse8': 'Unet6_512_ShuffleAttention10_mse_data2_e200_b8_lr001.log',
    # 'Unet6_512_ShuffleAttention11_mse8': 'Unet6_512_ShuffleAttention11_mse_data2_e200_b8_lr001.log',
    # 'cenet_mse': 'cenet_mse_data2_e200_b24_lr001.log',
    # 'segnet_mse':'segnet_mse_data2_e200_b24_lr001.log',
    # 'deeplabv3_mse': 'deeplabv3_mse_data2_e200_b24_0.001.log',
    # 'pspnet_mse': 'pspnet_mse_data2_e200_b24_lr001.log',
    # 'pspnet_mse_x': 'pspnet_mse_data2_e200_b24_lr001xnotaux.log',
    # 'Attunet_mse':'Attunet_mse_data2_e200_b12_lr001.log',

}

unet_512_logs = {
    'Unet6_512': 'Unet6_512_mse_data2_e200_b16_lr001.log',  # **
    'Unet_DAUNet3': 'Unet6_512_DAUNet3_mse_data2_e200_b8_lr001.log', # 这个好
    # 'Unet_CBAM3': 'Unet6_512_CBAM3_mse_data2_e200_b8_lr001.log',  # 这个
    'Unet_CBAM10': 'Unet6_512_CBAM10_mse_data2_e200_b8_lr001.log', #这个
    'Unet_PolarizedAttention3': 'Unet6_512_PolarizedAttention3_mse_data2_e200_b8_lr001.log', # 这个
    'Unet_PSA2_mse8': 'Unet6_512_PSA2_mse_data2_e200_b8_lr001.log', # 这个和9 差不多
    'Unet_SEAttention2': 'Unet6_512_SEAttention2_mse_data2_e200_b8_lr001.log', # 这个
    'Unet_CoTAttention7': 'Unet6_512_CoTAttention7_mse_data2_e200_b8_lr002.log', # 这个
    'Unet_ShuffleAttention3':'Unet6_512_ShuffleAttention3_mse_data2_e200_b8_lr001.log', # 这个
    'cenet': 'cenet_mse_data2_e200_b24_lr001.log',
    # 'segnet':'segnet_mse_data2_e200_b24_lr001.log',
    'deeplabv3': 'deeplabv3_mse_data2_e200_b24_0.001.log',
    # 'pspnet': 'pspnet_mse_data2_e200_b24_lr001.log',
    # 'pspnet_mse_x': 'pspnet_mse_data2_e200_b24_lr001xnotaux.log',
    'Attunet_mse':'Attunet_mse_data2_e200_b12_lr001.log',
}
def plot_multi_psnr_rmse2(filenames):
    fig = plt.figure(figsize=(18.5, 8))
    ndir = '/home/hb/work/pywork/ndvi2/UNet_NDVI/result/log'
    plt.title('validation rmse')
    for i in range(1, 10):
        ax1 = fig.add_subplot(3, 3, i)
        for k in filenames.keys():
            train_loss1, val_loss, val_rmse, val_psnr = get_loss(os.path.join(ndir, filenames[k]))
            x = [i for i in range(len(val_psnr))]
            val_psnr = smooth_curve(val_psnr, 0.1 * i)
            plt.plot(x, val_psnr, label=k, linewidth=4)
            plt.legend()
    plt.show()


if __name__ == '__main__':
    plt.rc('font', family='Times New Roman')
    plot_multi_train(unet_512_logs, 0.9)
    # plot_multi_psnr_rmse(unet_512_logs, 0.9)
#     # plot_ave()
#     # plot_total_loss()
#     # plot_multi_train(unet3_32_logs)
#     plot_multi_train(unet5_64_logs)
#     plot_multi_rmse(unet5_256_logs)
#     # plot_multi_val()
