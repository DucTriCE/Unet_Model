import torch
import cv2
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import os
import random
import math

def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """change color hue, saturation, value"""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

class MyDataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, transform=None ,valid=False):
        '''
        :param imList: image list (Note that these lists have been processed and pickled using the loadData.py)
        :param labelList: label list (Note that these lists have been processed and pickled using the loadData.py)
        :param transform: Type of transformation. SEe Transforms.py for supported transformations
        '''

        self.transform = transform
        self.Tensor = transforms.ToTensor()
        self.valid =valid
        if valid:
            self.root =r"data_2\\images\\val"
            self.names = os.listdir(self.root)
        else:
            self.root =r"data_2\\images\\train"
            self.names = os.listdir(self.root)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        '''

        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        W_ = 160
        H_ = 80
        image_name = os.path.join(self.root ,self.names[idx])
        image = cv2.imread(image_name)
        label1 = cv2.imread(image_name.replace("images","road"), 0)
        # if not os.path.isfile(label1):
        #     label1.replace()
        if not self.valid:
            if random.random() <0.5:
                augment_hsv(image)
            if random.random() < 0.5:
                image = np.fliplr(image)
                label1 = np.fliplr(label1)

        label1 = cv2.resize(label1, (W_, H_))
        image = cv2.resize(image, (W_, H_))

        _,seg_b1 = cv2.threshold(label1,1 ,255 ,cv2.THRESH_BINARY_INV)
        _,seg1 = cv2.threshold(label1,1 ,255 ,cv2.THRESH_BINARY)

        seg1 = self.Tensor(seg1)
        seg_b1 = self.Tensor(seg_b1)
        seg_da = torch.stack((seg_b1[0], seg1[0]) , 0)
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        return image_name ,torch.from_numpy(image) , seg_da