# -*- coding: utf-8 -*-
# @Time    : 2022/4/19 20:28
# @Author  : Tang
# @File    : datasets.py.py
# @Software: PyCharm
# 这是构建数据集代码
import glob
import os

import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms.Compose(transforms_)

        self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))

    def __getitem__(self, index):

        img_A = Image.open(self.files_A[index % len(self.files_A)]).convert('RGB')
        img_B = Image.open(self.files_B[index % len(self.files_B)]).convert('RGB')

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], 'RGB')
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], 'RGB')

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {'A': img_A, 'B': img_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))