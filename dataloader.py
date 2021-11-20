import json
import os
from collections import namedtuple

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import  matplotlib.pyplot as plt
import cv2
import h5py
import torch.utils.data as data


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

class FaceData(data.Dataset):

    def __init__(self, split='train', transform=None):
        self.transform = transform
        self.split = split
        self.data = h5py.File('./Facial_Expression_Data.h5', 'r', driver='core')
        if split=='train':
            self.imgs = self.data['Training_pixel']
            self.labels = self.data['Training_label']
            self.imgs = np.asarray(self.imgs)
            self.imgs = self.imgs.reshape((28709, 48, 48))

        elif split=='public_valid':
            self.imgs= self.data['PublicTest_pixel']
            self.labels = self.data['PublicTest_label']
            self.imgs = np.asarray(self.imgs)
            self.imgs = self.imgs.reshape((3589, 48, 48))

        else:
            self.imgs = self.data['PrivateTest_pixel']
            self.labels = self.data['PrivateTest_label']
            self.imgs = np.asarray(self.imgs)
            self.imgs = self.imgs.reshape((3589, 48, 48))

        print(self.__len__())

    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index]
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgs)



if __name__ == '__main__':
    import torchvision.transforms as transforms
    data = FaceData(split='public_valid',transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset=data,
                                               batch_size=1,
                                               shuffle=True, pin_memory=True, num_workers=1)
    print(data_loader.__len__())
    for i, input in enumerate(data_loader):
        # print(input[0])
        print(input[0].size())
        # exit()exit