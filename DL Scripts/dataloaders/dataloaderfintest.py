import glob
from this import s
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import pydicom
import pandas as pd
import pickle
from PIL import Image
import random
import os
import copy
import platform
import json
import os.path as osp
import re
import cv2
from .common import BaseDataset, Subset
from pdb import set_trace
import skimage
import torch.utils.data as data

random.seed(42432)

class GenericDataset(BaseDataset):
    def __init__(self, jsonPath, transforms, fix_length = None):

        super(GenericDataset, self).__init__(jsonPath)
        with open(jsonPath, 'r') as f :
            self.img_lists = json.load(f)     

        self.imgs_fin_list = self.clean_data()

        # floatType = torch.cuda.FloatTensor if useGPU else torch.FloatTensor
        # intType = torch.cuda.LongTensor if useGPU else torch.LongTensor

        random.shuffle(self.imgs_fin_list)
        self.transforms = transforms
        self.fix_length = fix_length

    def clean_data(self):
        imgs_fin_list = []
        for cd_ in self.img_lists:
            if osp.exists(cd_['img']):
                imgs_fin_list.append(cd_)
        
        return imgs_fin_list

    def preprocess_dicom_image(self, dicom_path):
        dicom = pydicom.read_file(dicom_path)
        image = dicom.pixel_array
        
        image = cv2.resize(image, (224, 224))
        clahe = cv2.createCLAHE(clipLimit=30.0, tileGridSize=(8,8))
        image = clahe.apply(image)

        image = image.astype(np.float32)
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255.0 # Convert to grayscale
        image = np.stack((image,)*3, axis=-1)  # Convert to 3 channel
        return image


    def __getitem__(self, index):
        index = index % len(self.imgs_fin_list)
        curr_dict = self.imgs_fin_list[index]
        # print('curr_dict:', curr_dict)

        image_path = curr_dict["img"]
        img_pp = self.preprocess_dicom_image(image_path)                                                                
        labels = int(curr_dict["label"])

        img_ = self.transforms(img_pp)
        
        return {'images': img_,
            'labels': labels,
            'path' : image_path
        }


    def __len__(self):

        if self.fix_length != None:
            assert self.fix_length >= len(self.imgs_fin_list)
            return self.fix_length
        else:
            return len(self.imgs_fin_list)


