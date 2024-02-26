import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 1er  class  de  reseau

class LiquideStateModel(nn.Module) :
    def __init__(self, input_size, hidden_size, output_size) :
        super(LiquideStateModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x) :
        _, (h_n, n) = self.lstm(x)

        out = self.fc(h_n.squeeze(0))
        return out


                # 2 : class d'attention et  elle est complete

class Attention(nn.Module) :
    def __init__(self, hidden_size) :
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x) :
        weights = torch.tanh(self.linear(x))
        weights = F.softmax(weights, dim=1)

        attended = torch.sum(weights * x, dim=1)
        return attended


# Attention   All***********

class Attention_All(nn.Module):
    def __init__(self,hidden_size):
        super(Attention_All,self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size,1)

    def forward(self,lstm_output):
        attention_scores = self.attention(lstm_output)
        attenmtion_scores =  attention_scores.squeeze(2)

        attention_weight = F.softmax(attenmtion_scores,dim=1).unsqueeze(2)
        weight_output = lstm_output * attention_weight
        context_vector = weight_output.sum(dim=1)

        return context_vector,attention_weight
#######################################

class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Sorties cachées et état caché du LSTM
        outputs, (h_n, c_n) = self.lstm(x)

        # Attention multi-têtes
        attention_weights = self.attention(outputs)

        # Contexte pondéré
        context = attention_weights * outputs

        # Classification
        logits = self.fc(context)

        return logits
        
# Multi  Head Attention ********
class MultiHeadAttention(nn.Module):
    def __init__(self,hidden_dim, num_heads):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim,hidden_dim)
        self.key = nn.Linear(hidden_dim,hidden_dim)
        self.value = nn.Linear(hidden_dim,hidden_dim)

        self.softmax = nn.Softmax(dim = -1)

    def forward(self,outputs):

        attention_scores = []
        for i in range(self.num_heads):
            query_i = self.query(outputs)
            key_i = self.key(outputs)
            value_i = self.value(outputs)

            attention_scores.append(torch.matmul(query_i,key_i.transpose(-1,-2)) / self.head_dim**0.5)

        attention_scores = torch.cat(attention_scores,dim=1)
        attention_weights = self.softmax(attention_scores)

        context = attention_weights * outputs

        return context








                #  3 class LSTM + attention   : complete
class Maestro_Model_Attention(nn.Module) :
    def __init__(self, input_size, hidden_size, num_layers, num_classes) :
        super(Maestro_Model_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size , num_classes)

    def forward(self, x) :
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        attended = self.attention(out)

        # out = (self.fc(out[:, -1, :]))
        # output = self.fc(attended)

        # Concaténation de la dernière sortie LSTM avec l'attention
        # Remarque : `.unsqueeze(1)` sur `attended` n'est peut-être pas nécessaire selon votre implémentation d'attention
        last_lstm_output = out[:, -1, :]  # Dernière sortie LSTM
        combined = torch.cat((last_lstm_output, attended), dim=1)  # Concaténation

        # Passage à travers la couche de sortie
        output = self.fc(combined)

        return output





#          4 class model  lstm*/*/*/*/**//***************
class Maestro_model_Lstm(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(Maestro_model_Lstm,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        #reseau RNN
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size,num_layers)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(x.device)
        c0 =  torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(x.device)

        #passage a travers du reseau
        out , _ = self.lstm(x,(h0,c0))

        #prendre   la  dernier  sorti
        out = out[:, -1,:]

        #passage a travers la couche sorti
        out = self.fc(out)

        return out


class Mastro_Attention(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(Mastro_Attention,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.attention = Attention_All(hidden_size)
        self.fc = nn.Linear(hidden_size,num_classes)


    def forward(self,x):
        lstm_out ,_ = self.lstm(x)
        context_vector, attention_weights = self.attention(lstm_out)
        output = self.fc(context_vector)


        return output,attention_weights





class Maestro_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(input_size,hidden_size)

        self.hidden_layer1 = nn.Linear(hidden_size,hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size,hidden_size)

        self.output_layer = nn.Linear(hidden_size,11)

        self.activation = nn.ReLU()
    def forward(self,x):
        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden_layer1(x))
        x = self.activation(self.hidden_layer2(x))
        x = self.output_layer(x)
        return x


class Instrument_Classifier(nn.Module):
    def __init__(self):
        super(Instrument_Classifier,self).__init__()

        self.fc1= nn.Linear(input_size,hidden_size)
        self.fc2= nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,out_features=11)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x













#Plotter  for  data visualizing



# parametre

input_size = 40
hidden_size = 128
num_layers = 2
# # instru_dataset = AudioDataset(r'C:\Users\y_mc\PycharmProjects\StromAI\Dataset\IRMAS-TrainingData')
#
# dataloader = DataLoader(instru_dataset, batch_size=16, shuffle=True)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam
