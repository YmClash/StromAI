import os

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
from mfcc_extracter import extract_mfcc,extract_mfcc_pytorch,extract_mfcc_LITE


sample_rate  = 44100
n_mfcc = 40
hop_length = 160
n_fft = 259
n_mels = 40
mel_scale = 'htk'


class AudioDataset(Dataset) :
    def __init__(self, directory, transform=None , sample_rate=sample_rate,n_mfcc=n_mfcc,log_mels=False) :
        self.directory = directory
        self.transform = transform
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.log_mels = log_mels
        self.hop_length = hop_length
        self.n_fft =n_fft
        self.mel_scale = mel_scale
        self.n_mels = n_mels

        self.files = []
        self.labels = []
        self.label_to_index = {}


        if not os.path.isdir(directory):
            raise  ValueError(f" le dossier specifie n'existe paa :{directory}")

        for label in os.listdir(directory) :
            label_dir = os.path.join(directory, label)
            if os.path.isdir(label_dir) :
                self.label_to_index[label] = len(self.label_to_index)
                for file in os.listdir(label_dir) :
                    if file.endswith('.wav') :
                        self.files.append(os.path.join(label_dir, file))
                        self.labels.append(self.label_to_index[label])

    def __len__(self) :
        ####Test##
        # print("LENGTH OF THE DATASET")
        # print(len(self.files))
        # ###########
        return len(self.files)

    def __getitem__(self, idx) :
        audio_path = self.files[idx]
        # label = self.labels[idx]
        label = os.path.basename(os.path.dirname(audio_path))
        waveform, _ = torchaudio.load(audio_path)

        if sample_rate != self.sample_rate :
            resample_transform = T.Resample(orig_freq=waveform.sample_rate, new_freq=self.sample_rate)
            waveform = resample_transform(waveform)

        # mfcc = extract_mfcc_pytorch(waveform, self.sample_rate, self.n_mfcc, self.hop_length, self.n_fft, self.mel_scale)

        mfcc = extract_mfcc_LITE(waveform, self.sample_rate, self.n_mfcc)




        # if self.transform:
        #     mfcc = self.transform(mfcc)
        # label_tensor = label
        # ####test##
        # print("WAVEFORM AND LABEL")
        # print(mfcc)
        # print(label)
        ##########

        return mfcc, label

# #
# TRAIN_DATA_DIR = r"C:\Users\y_mc\PycharmProjects\StromAI\Dataset\IRMAS-TrainingData"
# #
# train_data = AudioDataset(TRAIN_DATA_DIR)
# train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
# print(len(train_data))
# #
# # Chargement et utilisation d'un élément
# for waveform, label in train_loader:
#     mfcc = extract_mfcc_LITE(waveform, sr=sample_rate, n_mfcc=n_mfcc)
#     print(f"Waveform shape: {mfcc.shape}, Label: {label}")
#
# # for mfcc,label in train_loader:
# #     print(f"Waveform shape: {mfcc.shape}:{mfcc}, Label: {label}")
# #

