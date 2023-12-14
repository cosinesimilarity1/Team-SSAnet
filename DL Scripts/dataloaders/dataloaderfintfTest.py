import glob
from this import s
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import pydicom
import torchvision.models as models
import pandas as pd
import pickle
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import torch.nn.functional as F
import random
import os
import copy
import platform
from torch import nn
import json
import os.path as osp
import re
import cv2
from .common import BaseDataset, Subset
from pdb import set_trace
import skimage
import torch.utils.data as data
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

random.seed(42432)

class GenericDataset(BaseDataset):
    def __init__(self, csv, transforms, featExtractor, fix_length = None):

        super(GenericDataset, self).__init__(csv)

        self.data_df = pd.read_csv(csv)
        self.data_df['anon_dicom_path'] = self.data_df['anon_dicom_path'].apply(self.convert_to_list)
        self.data_df['anon_dicom_path'] = self.data_df['anon_dicom_path'].apply(self.replace_path_prefix)
        self.label_assignment()    

        self.transforms = transforms
        self.featExtractor = featExtractor

        self.fix_length = fix_length

    def label_assignment(self):
        self.data_df = shuffle(self.data_df)
        label_encoder = LabelEncoder()
        self.data_df['asses'] = label_encoder.fit_transform(self.data_df['asses'])    


    def convert_to_list(self, string):
        return string.strip("[]").replace("'", "").split(", ")

    # Function to replace path prefix in file paths
    def replace_path_prefix(self, paths, cohort_keywords=['/cohort_1', 'cohort_2']):
        new_prefix = '/local/scratch/shared-directories/ssanet/embed-dataset-aws/images/'
        updated_paths = []
        for path in paths:
            for keyword in cohort_keywords:
                if keyword in path:
                    split_path = path.split(keyword)
                    updated_path = new_prefix + keyword.lstrip('/') + split_path[1]
                    updated_paths.append(updated_path)
                    break
            else:
                updated_paths.append(path)
        return updated_paths        

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
        index = index % len(self.data_df)
        curr_df = self.data_df.iloc[index]

        image_paths = curr_df["anon_dicom_path"]
        acc_non = curr_df["acc_anon"]
        
        images = []
        for image_path_ in image_paths:
            img_t = self.transforms(self.preprocess_dicom_image(image_path_))
            images.append(torch.unsqueeze(img_t, 0))
        
        img_pp = torch.cat(images)

        labels = int(curr_df["asses"])

        img_ = img_pp

        feats = self.featExtractor(img_)
        feats = feats.squeeze(-1).squeeze(-1)
        
        return {'feats': feats,
            'labels': labels,
            'acc_anon' : acc_non
        }


    def __len__(self):

        if self.fix_length != None:
            assert self.fix_length >= len(self.data_df)
            return self.fix_length
        else:
            return len(self.data_df)
