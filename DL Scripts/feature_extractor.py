import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataloaderfin import GenericDataset
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
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
import skimage
from sklearn.metrics import f1_score, fbeta_score
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
import GPUtil
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.multiprocessing as mp


def mp_fn(local_rank, *args):
    dist.init_process_group("nccl",
                            rank=local_rank,
                            world_size=torch.cuda.device_count())
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    jsonTrain = 'train_sample.json'
    jsonVal = 'valid_sample.json'

    train_df = pd.read_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/train.csv')

    trainTransform = transforms.Compose([transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5)])

    validTransform = transforms.Compose([transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5)])
            
    tr_parent = GenericDataset(jsonTrain,
                                trainTransform)

    val_parent = GenericDataset(jsonVal,
                                validTransform)                            


    trainloader = DataLoader(
            tr_parent,
            batch_size=128,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            drop_last=True
        )     

    valloader = DataLoader(
            val_parent,
            batch_size= 128,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            drop_last=True
        )            


    # # Load the pre-trained ResNet50 model
    model_resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    num_ftrs = model_resnet50.fc.in_features
    model_resnet50.fc = nn.Linear(num_ftrs, 4)

    # Define device, loss, and optimizer
    model_resnet50 = model_resnet50.to(device)
    model_resnet50 = DDP(model_resnet50, device_ids=[local_rank])
    class_weights = dict(enumerate(train_df['asses'].value_counts().sort_index() / len(train_df['asses'])))
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(list(class_weights.values())).to(device))
    optimizer = optim.Adam(model_resnet50.parameters(), lr=0.001)

    print(torch.cuda.current_device())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))


    # t0 = time.perf_counter()
    # summ = 0
    # count = 0

    # Training loop
    best_val_acc = 0
    best_model_path = None
    for epoch in range(25):
        print("******************************************************************************")
        print('################################## TRAINING ##################################')
        print("******************************************************************************")
        print(f"Epoch {epoch}")
        model_resnet50.train()
        running_loss = 0.0
        for step, data in enumerate(trainloader, 0):
            inputs, labels = data["images"].to(device=device, non_blocking=True), data["labels"].to(device=device, non_blocking=True)
            # print(inputs.shape, labels.shape)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model_resnet50(inputs)
                loss = criterion(outputs, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step % 200 == 199:
                print(f"[{epoch + 1}, {step + 1}] loss: {running_loss / 200:.3f}")
                running_loss = 0.0

        epoch_model_path = '/local/scratch/shared-directories/ssanet/SCRIPTS/models_sample/model_' + str(epoch) + '.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_resnet50.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, epoch_model_path)

        
        # Validate after each epoch
        print("********************************************************************************")
        print('################################## VALIDATING ##################################')
        print("********************************************************************************")
        model_resnet50.eval()
        val_accuracy = 0
        with torch.no_grad():
            for i, data in enumerate(valloader, 0):
                inputs, labels = data["images"], data["labels"]
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print(i, inputs.shape, labels.shape, flush = True)                
                outputs = model_resnet50(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_accuracy += accuracy_score(labels.cpu(), predicted.cpu())
        
        val_accuracy /= len(valloader)
        print("VAL ACCURACY @ EPOCH " + str(epoch) + " :", val_accuracy)
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_path = '/local/scratch/shared-directories/ssanet/SCRIPTS/models_sample/best_model_' + str(epoch) + '.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_resnet50.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_val_acc,
            }, best_model_path)

        # batch_time = time.perf_counter() - t0
        # if step > 10:  # skip first steps
        #     summ += batch_time
        #     count += 1
        # t0 = time.perf_counter()
        # if step > 50:
        #     break

    print(f'average step time: {summ/count}')
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    print('Finished Training')

    print("Saving final model")
    torch.save(model_resnet50.state_dict(), '/local/scratch/shared-directories/ssanet/SCRIPTS/models_sample/resnet50_final_epoch_model.pt')

if __name__ == '__main__':
    mp.spawn(mp_fn,
             args=(),
             nprocs=torch.cuda.device_count(),
             join=True)