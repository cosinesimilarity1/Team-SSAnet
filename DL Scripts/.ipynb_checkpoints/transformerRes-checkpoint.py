import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataloaderfintf import GenericDataset
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, fbeta_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import nvidia_smi
import torch.nn.functional as F
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
# import torch.multiprocessing as mp
# mp.set_sharing_strategy('file_system')

class ResNet50Extractor(nn.Module):
    def __init__(self, resnet50):
        super(ResNet50Extractor, self).__init__()
        self.resnet50 = resnet50
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])

    def forward(self, x):
        return self.feature_extractor(x)

class TransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, num_classes):
        super(TransformerClassifier, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x = pack_padded_sequence(x.data, lengths, batch_first=True, enforce_sorted=False)
        x = self.transformer(x)
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x, _ = pad_packed_sequence(x)
        x = F.avg_pool1d(x.transpose(1, 2), kernel_size=x.size(1)).squeeze(-1)
        x = self.fc(x)
        return x   

def my_collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x), reverse=True)
    sequences = [item['feats'] for item in batch]
    lengths = [len(seq) for seq in sequences]
    lengths_tensor = torch.tensor(lengths)
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    labels = torch.tensor([item['labels'] for item in batch])

    return {'input_sequence': padded_sequences, 'lengths': lengths_tensor, 'labels': labels}


torch.cuda.set_device(0)
device = torch.cuda.current_device()

model_resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model_resnet50.fc.in_features
model_resnet50.fc = nn.Linear(num_ftrs, 4)

print("LOADING THE BEST PERFORMING RESNET50 MODEL FOR EVALUATION ON TEST DATASET from following path:")
best_model_path = '/local/scratch/shared-directories/ssanet/SCRIPTS/models_sample/best_model_16.pt'
print(best_model_path)
checkpoint = torch.load(best_model_path, map_location='cuda:0')

prefix = 'module.'
n_clip = len(prefix)
adapted_dict = {k[n_clip:]: v for k, v in checkpoint['model_state_dict'].items()
                if k.startswith(prefix)}
model_resnet50.load_state_dict(adapted_dict)

for param in model_resnet50.parameters():
    param.requires_grad = False

resnet_extractor = ResNet50Extractor(model_resnet50)

csvTrain = 'train_clean.csv'
csvVal = 'val.csv'

train_df = pd.read_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/train_clean.csv')

trainTransform = transforms.Compose([transforms.ToTensor(),
transforms.RandomHorizontalFlip(p=0.5)])

validTransform = transforms.Compose([transforms.ToTensor(),
transforms.RandomHorizontalFlip(p=0.5)])

tr_parent = GenericDataset(csvTrain, trainTransform, resnet_extractor)

val_parent = GenericDataset(csvVal, validTransform, resnet_extractor)                            

batch_size = 128

trainloader = DataLoader(
        tr_parent,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        drop_last=False,
        collate_fn = my_collate_fn
    )     

valloader = DataLoader(
        val_parent,
        batch_size= batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        drop_last=False,
        collate_fn = my_collate_fn
    )            


input_size_tf = 2048
num_classes = 4

model_tf = TransformerClassifier(input_size=input_size_tf, 
                                hidden_size=2048, 
                                num_layers=2, num_heads=4, 
                                num_classes=num_classes)

model_tf = model_tf.to(device)
class_weights = dict(enumerate(train_df['asses'].value_counts().sort_index() / len(train_df['asses'])))
criterion = nn.CrossEntropyLoss(weight=torch.Tensor(list(class_weights.values())).to(device))
optimizer = optim.Adam(model_tf.parameters(), lr=0.001)

print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# Training loop
best_val_acc = 0
best_model_path = None
for epoch in range(25):
    print("******************************************************************************")
    print('################################## TRAINING ##################################')
    print("******************************************************************************")
    print(f"Epoch {epoch}")
    model_tf.train()
    running_loss = 0.0
    for step, data in enumerate(trainloader, 0):
        print(step)
        inputs, labels, lengths = data["input_sequence"].to(device=device, non_blocking=True), data["labels"].to(device=device, non_blocking=True), data['lengths']
        outputs = model_tf(inputs, lengths)
        loss = criterion(outputs, labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if step % 200 == 199:
            print(f"[{epoch + 1}, {step + 1}] loss: {running_loss / 200:.3f}")
            running_loss = 0.0            

    epoch_model_path = '/local/scratch/shared-directories/ssanet/SCRIPTS/models_sample/model_tf_' + str(epoch) + '.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_tf.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, epoch_model_path)

    
    # Validate after each epoch
    print("********************************************************************************")
    print('################################## VALIDATING ##################################')
    print("********************************************************************************")
    model_tf.eval()
    val_accuracy = 0
    with torch.no_grad():
        for i, data in enumerate(valloader, 0):
            inputs, labels, lengths = data["input_sequence"].to(device=device, non_blocking=True), data["labels"].to(device=device, non_blocking=True), data['lengths']
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_tf(inputs, lengths)
            _, predicted = torch.max(outputs.data, 1)
            val_accuracy += accuracy_score(labels.cpu(), predicted.cpu())
    
    val_accuracy /= len(valloader)
    print("VAL ACCURACY @ EPOCH " + str(epoch) + " :", val_accuracy)
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        best_model_path = '/local/scratch/shared-directories/ssanet/SCRIPTS/models_sample/best_model_tf_' + str(epoch) + '.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_tf.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': best_val_acc,
        }, best_model_path)


print(f'average step time: {summ/count}')
print(f"Validation Accuracy: {val_accuracy:.4f}")

print('Finished Training')

print("Saving final model")
torch.save(model_tf.state_dict(), '/local/scratch/shared-directories/ssanet/SCRIPTS/models_sample/transformer_final_epoch_model.pt')
