import torch
import cv2
# import os
# import glob
from torch.utils.data import Dataset
import torchvision.transforms as Transforms
import numpy as np
import random
import scipy.io as io
from skimage.transform import resize
# from PIL import Image
import argparse


def to_standardize(image):
    mean = np.mean(image)
    var = np.mean(np.square(image - mean))
    image = (image - mean) / np.sqrt(var)
    return image


def to_normalization(image):
    image_max = np.max(image)
    image_min = np.min(image)
    image = (image - image_min) / (image_max - image_min)
    return image


def idx2coordinates(idx, col):
    i = idx // col
    j = idx % col
    return i, j


def Load_data(data_name, t1_path, t2_path, gt_path, patch_size, mode):
    ps = patch_size
    w = 0
    h = 0
    c = 0
    if data_name == "bastrop":
        mat = io.loadmat(t1_path)
        t1 = mat['t1_L5'][:, :, 3]
        t2 = mat["t2_ALI"][:, :, 5]
        t1 = to_normalization(t1)
        t2 = to_normalization(t2)
        gt = mat["ROI_1"]
    elif data_name == "california_mat":
        mat = io.loadmat(t1_path)
        t1 = mat['image_t1'][:, :, 0]
        t2 = mat["image_t2"][:, :, 3]
        t1 = to_normalization(t1)
        t2 = to_normalization(t2)
        gt = mat["gt"]
    elif data_name == "california":
        t1 = cv2.imread(t1_path)
        t2 = cv2.imread(t2_path)
        gt = cv2.imread(gt_path)[:, :, 0]
        # t1 = to_normalization(t1)
        # t2 = to_normalization(t2)
        # gt = mat["gt"]
    elif data_name == "italy":
        t1 = cv2.imread(t1_path)
        t2 = cv2.imread(t2_path)
        gt = cv2.imread(gt_path)[:, :, 0]
    elif data_name == "yellow":
        t1 = cv2.imread(t1_path)[:, :, 0]
        t2 = cv2.imread(t2_path)[:, :, 0]
        gt = cv2.imread(gt_path)[:, :, 0]
    elif data_name == "shuguang":
        t1 = cv2.imread(t1_path)
        t2 = cv2.imread(t2_path)
        gt = cv2.imread(gt_path)[:, :, 0]
    elif data_name == "gloucester2":
        t1 = cv2.imread(t1_path)
        t2 = cv2.imread(t2_path)
        gt = cv2.imread(gt_path)[:, :, 0]
    elif data_name == "gloucester1":
        t1 = cv2.imread(t1_path)
        t2 = cv2.imread(t2_path)
        gt = cv2.imread(gt_path)[:, :, 0]
        # [oh, ow] = gt.shape
        # t1 = resize(img1, (oh // 4, ow // 4), mode='constant', preserve_range=True)
        # t2 = resize(img2, (oh // 4, ow // 4), mode='constant', preserve_range=True)
        # gt = resize(imggt, (oh // 4, ow // 4), mode='constant', preserve_range=True)
    elif data_name == "france":
        t1 = cv2.imread(t1_path)
        t2 = cv2.imread(t2_path)
        gt = cv2.imread(gt_path)[:, :, 0]

    if t1.ndim == 2 or t2.ndim == 2:
        if t1.ndim == 2:
            t1 = t1[..., np.newaxis]
        if t2.ndim == 2:
            t2 = t2[..., np.newaxis]

    # gt = np.squeeze(gt)
    # t1 = to_normalization(t1)
    # t2 = to_normalization(t2)
    [w, h, c] = t1.shape
    t1_expand = cv2.copyMakeBorder(t1, ps // 2, ps // 2, ps // 2, ps // 2, cv2.BORDER_DEFAULT)
    t2_expand = cv2.copyMakeBorder(t2, ps // 2, ps // 2, ps // 2, ps // 2, cv2.BORDER_DEFAULT)
    # t1_std = to_standardize(t1_expand).astype(np.float32)
    # t2_std = to_standardize(t2_expand).astype(np.float32)
    t1_list = []
    t2_list = []
    gt_list = []
    if mode == 'train':
        for i in range(0, w, ps):
            for j in range(0, h, ps):
                # for i in range(w):
                #     for j in range(h):
                t1_list.append(t1_expand[i:i + ps, j:j + ps])
                t2_list.append(t2_expand[i:i + ps, j:j + ps])
                gt_list.append(gt[i, j] / 255)
    elif mode == 'test':
        for i in range(w):
            for j in range(h):
                t1_list.append(t1_expand[i:i + ps, j:j + ps])
                t2_list.append(t2_expand[i:i + ps, j:j + ps])
                gt_list.append(gt[i, j] / 255)

    return t1_list, t2_list, gt_list


class Data_Loader(Dataset):
    def __init__(self, data_name, t1_path, t2_path, gt_path, patch_size, mode, transform=None):
        """
        -----Init Param settings-----
        data_name: "Bastrop" or "California" or other
        t1_path: t1 image path
        t2_path: t2 image path
        gt_path: gt image path
        patch_size: default 7
        """
        self.data_name = data_name
        self.t1_path = t1_path
        self.t2_path = t2_path
        self.gt_path = gt_path
        self.ps = patch_size
        self.mode = mode
        self.transform = transform
        self.t1, self.t2, self.gt = Load_data(self.data_name, self.t1_path, self.t2_path, self.gt_path, self.ps,
                                              self.mode)

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, intex):
        # 根据index读取图像
        # 随机进行数据增强，为2时不处理
        # flipCote = random.choice([-1, 0, 1, 2])
        # if flipCote != 2:
        #     image1 = self.augment(image1, flipCote)
        #     image2 = self.augment(image2, flipCote)
        #     label = self.augment(label, flipCote)

        # label = label.reshape(label.shape[0], label.shape[1], 1)
        patch1 = self.transform(self.t1[intex])
        patch2 = self.transform(self.t2[intex])
        label = torch.tensor(self.gt[intex])

        return patch1, patch2, label

    def __len__(self):
        return len(self.t1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='other', type=str)  # Bastrop or California or other
    parser.add_argument('--t1_path', default='./data/Italy/Italy_1.bmp', type=str)
    parser.add_argument('--t2_path', default='./data/Italy/Italy_2.bmp', type=str)
    parser.add_argument('--gt_path', default='./data/Italy/Italy_gt.bmp', type=str)
    parser.add_argument('--patch_size', default=3, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    args = parser.parse_args()

    isbi_dataset = Data_Loader(data_name=args.data_name, t1_path=args.t1_path,
                               t2_path=args.t2_path, gt_path=args.gt_path,
                               patch_size=args.patch_size, transform=Transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False)
    i = 0
    for image1, image2, label in train_loader:
        print(i)
        i += 1
