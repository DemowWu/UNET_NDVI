import torch.utils.data as data
import os
import numpy as np
import torch
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt


# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#
#
# def get_metrics(mask_name, predict, val_loss):
#     image_mask = np.load(mask_name)
#     # loss = val_loss(predict, image_mask)
#     loss = None
#     mse = mean_squared_error(image_mask.flatten(), predict.flatten())
#     mae = mean_absolute_error(image_mask.flatten(), predict.flatten())  # 平均绝对误差
#     r2 = r2_score(image_mask.flatten(), predict.flatten())
#     # error = torch.abs(image_mask - predict).sum().data
#     # squared_error = ((image_mask - predict) * (image_mask - predict)).sum().data
#     # mse = math.sqrt(running_mse\len(loader_test))
#     # mae = running_mae\len(loader_test)
#     # runnning_mae += error
#     # runnning_mse += squared_error
#
#     return mse, mae, r2, loss


class NdviDataset(data.Dataset):
    def __init__(self, root, files, is_aug, device,transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.files = files
        self.pics, self.masks = self.getDataPath()
        self.device=device
        self.img_aug = iaa.OneOf([ # 其一增强器：给出的数据随机选择一个做增强
            iaa.AdditiveGaussianNoise(scale=0.2), # 高斯噪声增强器:添加高斯白噪声到图象
            iaa.GaussianBlur(sigma=(0, 3.0)), # 高斯模糊增强器
            iaa.AverageBlur(k=(2, 7)), # 平均模糊增强器
            iaa.Sharpen(alpha=0.5), # 锐化增强器
            iaa.AdditiveLaplaceNoise(scale=(0, 0.2)), # 拉普拉斯噪声增强器：添加拉普拉斯噪声到图像
            iaa.AdditivePoissonNoise(lam=(0, 0.2)), # 泊松噪声增强器：添加泊松噪声到图像
            iaa.Identity(), # 不改变输入数据的增强器
            iaa.Identity()
        ])
        self.is_aug = is_aug # 增强器判断，true——增强操作

    def getDataPath(self):
        pics = []
        masks = []
        for i in self.files:
            img = os.path.join(self.root, "{}.npy".format(i))
            mask = os.path.join(self.root, "{}_mask.npy".format(i)) # 掩膜处理
            pics.append(img)
            masks.append(mask)

        return pics, masks

    def __getitem__(self, index):
        x_path = self.pics[index]
        y_path = self.masks[index]
        origin_x = np.load(x_path)
        origin_y = np.load(y_path)
        # plt.imshow(origin_x)
        # plt.show()
        if self.is_aug:
            origin_x = self.img_aug(images=origin_x)
        origin_x = self.normalization(origin_x)
        # plt.imshow(origin_x)
        # plt.show()
        # if self.target_transform is not None:
        origin_y = self.normalization(origin_y)
        if self.transform is not None:
            # origin_x = self.transform(origin_x)
            origin_x = torch.Tensor(origin_x)
            origin_x = origin_x.unsqueeze(0) # 数据维度扩充，0——行方向扩充:[1,2,3,4]——>[[1,2,3,4]];1——列方向扩充：[1,2,3,4]——>[[1],[2],[3],[4]]

        if self.target_transform is not None:
            # origin_y = self.target_transform(origin_y)
            origin_y = torch.Tensor(origin_y)
            origin_y = origin_y.unsqueeze(0)

        return origin_x, origin_y, x_path, y_path

    def __len__(self):
        return len(self.pics)

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
