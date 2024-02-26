import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from model import Mastro_Attention
from train import AudioDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 40
hidden_size = 128
num_layers = 2
num_classes = 11

DATASET_DIR = (r'C:\Users\y_mc\PycharmProjects\StromAI\Dataset\IRMAS-TrainingData')

train_data = AudioDataset(DATASET_DIR)

dataset_size = len(train_data)
train_size = int(dataset_size * 0.8)
test_size = dataset_size - train_size

train_dataset, test_dataset = random_split(train_data, [train_size, test_size])

# print(dataset_size)
# print(train_size)
# print(test_size)
# print(len(train_dataset))
# print(len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print()
print(len(train_loader))
print(len(test_loader))


# parameter

learn_rate = 0.001
num_epochs = 5


maestro = Mastro_Attention(input_size=input_size, hidden_size=hidden_size, num_layers=2, num_classes=11)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(maestro.parameters(),lr=learn_rate)
maestro = maestro.to(device)


print("Startin Training ")
print(f'Device : {device}')
for epoch in range(num_epochs):
    for inputs, labels in train_loader :
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = maestro(inputs)
        if isinstance(outputs,tuple):
            outputs = outputs[0]
        loss = criterion(outputs,labels)

        loss.backward()
        optimizer.step()


    print(f'Epoque [{epoch+1}/{num_epochs}],Perte : {loss.item():.4f}')


print("Done.....")


