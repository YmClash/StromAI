import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,random_split
import torch.nn.functional as F
from model import Mastro_Attention
from train import AudioDataset

input_size = 40
hidden_size = 128
num_layers = 2
num_classes = 11

DATASET_DIR =(r'C:\Users\y_mc\PycharmProjects\StromAI\Dataset\IRMAS-TrainingData')


train_data =AudioDataset(DATASET_DIR)

dataset_size = len(train_data)
train_size = int(dataset_size * 0.8)
test_size = dataset_size -train_size

train_dataset ,test_dataset = random_split(train_data,[train_size,test_size])

print(dataset_size)
print(train_size)
print(test_size)
print(len(train_dataset))
print(len(test_dataset))

train_loader = DataLoader(train_dataset,batch_size=32 ,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)

criterion = nn.CrossEntropyLoss
optimizer = optim.Adam()

