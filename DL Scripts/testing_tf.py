import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataloaderfintfTest import GenericDataset
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, fbeta_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import nvidia_smi
import time
import torch.nn.functional as F
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
        # self.embedding = nn.Embedding(input_size, hidden_size)
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
    # Sort the batch by sequence length in descending order
    batch = sorted(batch, key=lambda x: len(x), reverse=True)
    
    # Create a list of sequences and their lengths
    sequences = [item['feats'] for item in batch]
    lengths = [len(seq) for seq in sequences]

    lengths_tensor = torch.tensor(lengths)
    
    # Pad sequences to the length of the longest sequence in the batch
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

    labels = torch.tensor([item['labels'] for item in batch])

    acc_anons = torch.tensor([item['acc_anon'] for item in batch])

    return {'input_sequence': padded_sequences, 'lengths': lengths_tensor, 'labels': labels, 'acc_anon' : acc_anons}



device = torch.device('cpu')
# torch.cuda.set_device(0)
# device = torch.cuda.current_device()

model_resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model_resnet50.fc.in_features
model_resnet50.fc = nn.Linear(num_ftrs, 4)

print("LOADING THE BEST PERFORMING RESNET50 MODEL FOR EVALUATION ON TEST DATASET from following path:")
best_model_path = '/local/scratch/shared-directories/ssanet/SCRIPTS/models_sample/best_model_16.pt'
print(best_model_path)
checkpoint = torch.load(best_model_path, map_location = device)

prefix = 'module.'
n_clip = len(prefix)
adapted_dict = {k[n_clip:]: v for k, v in checkpoint['model_state_dict'].items()
                if k.startswith(prefix)}
model_resnet50.load_state_dict(adapted_dict)

for param in model_resnet50.parameters():
    param.requires_grad = False

resnet_extractor = ResNet50Extractor(model_resnet50)    

input_size_tf = 2048
num_classes = 4

model_tf = TransformerClassifier(input_size=input_size_tf, 
                                hidden_size=2048, 
                                num_layers=2, num_heads=4, 
                                num_classes=num_classes)

best_model_tf_path = '/local/scratch/shared-directories/ssanet/SCRIPTS/models_sample/model_tf_15.pt'                                
checkpoint_tf = torch.load(best_model_tf_path, map_location = device)

model_tf.load_state_dict(checkpoint_tf['model_state_dict'])

csvTest = 'test.csv'

testTransform = transforms.Compose([transforms.ToTensor(),
transforms.RandomHorizontalFlip(p=0.5)])

test_parent = GenericDataset(csvTest, testTransform, resnet_extractor)                            

batch_size = 1

testloader = DataLoader(
        test_parent,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
        collate_fn = my_collate_fn
    )     

print('testloader:', len(testloader))

y_pred = []
y_true = []
y_pred_probs = []
output_df_data = []
print("******************************************************************************")
print('################################## TESTING ##################################')
print("******************************************************************************")
start = time.time()
model_tf.eval()
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        print(i)
        inputs, labels, lengths, acc_anon = data["input_sequence"].to(device=device, non_blocking=True), data["labels"].to(device=device, non_blocking=True), data['lengths'], data['acc_anon']
        # print('labels:', labels)
        inputs = inputs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        outputs = model_tf(inputs, lengths)
        _, predicted = torch.max(outputs.data, 1)
        m = nn.Softmax(dim=1)
        prob = m(outputs)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        y_pred_probs.append(list(prob.cpu().numpy()[0]))
        output_df_data.append({'acc_anon' : acc_anon, 'GT' : labels.cpu().numpy(), 'PRED' : predicted.cpu().numpy(), 'PROB' : list(prob.cpu().numpy()[0])})


with open('testoutputep16_tf.pkl', 'wb') as handle:
    pickle.dump(output_df_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("TIME TAKEN:", time.time() - start)

print('y_true:', y_true)
print('y_pred_probs:', y_pred_probs)

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
