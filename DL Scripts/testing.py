import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataloaderfintest import GenericDataset
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, fbeta_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import nvidia_smi
import time
import cv2
from pathlib import Path
import pydicom
import numpy as np
from PIL import Image
import time
import skimage
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
import GPUtil
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')


# def mp_fn(local_rank, *args):
#     dist.init_process_group("nccl",
#                             rank=local_rank,
#                             world_size=torch.cuda.device_count())
# torch.cuda.set_device(torch.cuda.current_device())
# device = torch.cuda.current_device()
device = torch.device('cpu')

    # jsonTrain = 'train_sample.json'
    # jsonVal = 'valid_sample.json'
jsonTest = 'test_sample.json'
    # train_df = pd.read_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/train.csv')

    # trainTransform = transforms.Compose([transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(p=0.5)])

    # validTransform = transforms.Compose([transforms.ToTensor(),
    # transforms.RandomHorizontalFlip(p=0.5)])

testTransform = transforms.Compose([transforms.ToTensor(),
transforms.RandomHorizontalFlip(p=0.5)])    
            
    # tr_parent = GenericDataset(jsonTrain,
    #                             trainTransform)

    # val_parent = GenericDataset(jsonVal,
    #                             validTransform)      

test_parent = GenericDataset(jsonTest,
                            testTransform)                                                            


    # trainloader = DataLoader(
    #         tr_parent,
    #         batch_size=128,
    #         shuffle=True,
    #         num_workers=16,
    #         pin_memory=True,
    #         drop_last=True
    #     )     

    # valloader = DataLoader(
    #         val_parent,
    #         batch_size= 128,
    #         shuffle=True,
    #         num_workers=16,
    #         pin_memory=True,
    #         drop_last=True
    #     )            

testloader = DataLoader(
    test_parent,
    batch_size= 1,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    drop_last=True
)                    

print('testloader:', len(testloader))

model_resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model_resnet50.fc.in_features
model_resnet50.fc = nn.Linear(num_ftrs, 4)

print("LOADING THE BEST PERFORMING RESNET50 MODEL FOR EVALUATION ON TEST DATASET from following path:")
best_model_path = '/local/scratch/shared-directories/ssanet/SCRIPTS/models_sample/best_model_16.pt'
print(best_model_path)
checkpoint = torch.load(best_model_path, map_location=device)

prefix = 'module.'
n_clip = len(prefix)
adapted_dict = {k[n_clip:]: v for k, v in checkpoint['model_state_dict'].items()
                if k.startswith(prefix)}
model_resnet50.load_state_dict(adapted_dict)

# model_resnet50.load_state_dict(torch.load(best_model_path))
model_resnet50.eval()


y_pred = []
y_true = []
y_pred_probs = []
output_df_data = []
y_true_ohe = []
print("******************************************************************************")
print('################################## TESTING ##################################')
print("******************************************************************************")
start = time.time()
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        if i % 100 == 0:
            print(i)
        inputs, labels, paths = data['images'], data['labels'], data['path']
        # print('labels:', labels)
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_resnet50(inputs)
        _, predicted = torch.max(outputs.data, 1)
        m = nn.Softmax(dim=1)
        prob = m(outputs)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        y_pred_probs.append(list(prob.cpu().numpy()[0]))
        output_df_data.append({'paths' : paths, 'GT' : labels.cpu().numpy(), 'PRED' : predicted.cpu().numpy(), 'PROB' : list(prob.cpu().numpy()[0])})


with open('testoutput16_res.pkl', 'wb') as handle:
	pickle.dump(output_df_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("TIME TAKEN:", time.time() - start)


print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
f1_micro = f1_score(y_true, y_pred, average='micro')
f1_macro = f1_score(y_true, y_pred, average='macro')
f2_micro = fbeta_score(y_true, y_pred, beta=2, average='micro')
f2_macro = fbeta_score(y_true, y_pred, beta=2, average='macro')
prec = precision_score(y_true, y_pred, average='macro')
recall_ = recall_score(y_true, y_pred, average='macro')
roc_auc_ = roc_auc_score(np.array(y_true), np.array(y_pred_probs), multi_class='ovr')
print(f"F1 micro: {f1_micro}")
print(f"F1 macro: {f1_macro}")
print(f"F2 micro: {f2_micro}")
print(f"F2 macro: {f2_macro}")
print(f"Precision: {prec}")
print(f"Recall: {recall_}")
print(f"ROC_AUC: {roc_auc_}")

# Additional metrics can be added as per requirement
