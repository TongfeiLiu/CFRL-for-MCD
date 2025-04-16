import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import torchvision.transforms as Transforms

class ISBI_Loader(Dataset):
    def __init__(self, data_path, transform=None):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.tif'))
        self.transform = transform

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, intex):
        # 根据index读取图像
        image_path = self.imgs_path[intex]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label')
        # 读取训练图像和标签
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # 将图像转为单通道图片
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = label.reshape(label.shape[0], label.shape[1], 1)
        image = self.transform(image)
        label = self.transform(label)
        """
        # 随机进行数据增强，为2时不处理
        flipCote = random.choice([-1, 0, 1, 2])
        if flipCote != 2:
            image = self.augment(image, flipCote)
            label = self.augment(label, flipCote)
        """
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

if __name__ == "__main__":
    isbi_dataset = ISBI_Loader(data_path="C:\\Users\\Administrator\\Desktop\\U-Net\\data\\AriealData\\train\\",
                               transform=Transforms.ToTensor())
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=4,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)