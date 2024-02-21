import librosa
import torch
import torchaudio
from torchaudio.transforms import MFCC
from torchaudio import load
import os
from mfcc_extracter import extract_mfcc,extract_mfcc_LITE,extract_mfcc_pytorch
from visual import plot_waveform,plot_spectrogram,plot_fbank,plot_pitch
from train import AudioDataset
# import matplotlib.pyplot as plt
from train import AudioDataset
from torch.utils.data import Dataset, DataLoader




PIANO_SAMPLE = r"C:\Users\y_mc\PycharmProjects\StromAI\Dataset\IRMAS-TrainingData\pia\[pia][cla]1293__2.wav"
TEST_SET_DIR = r"C:\Users\y_mc\PycharmProjects\StromAI\Dataset\IRMAS-TrainingData"


# Test Output

sample_rate = 44100
n_mfcc = 40

n_fft = 400
hop_length = 160
n_mels = 40
mel_scale = "htk"

# # extract data function testing
# dataset = AudioDataset(TEST_SET_DIR)
# print(dataset)
# # data = DataLoader(dataset,batch_size=32,shuffle=True)

# Dataset_1 = extract_mfcc(TEST_SET_DIR,sr=sample_rate,n_mfcc=n_mfcc)
# Dataset_2 = extract_mfcc_LITE(PIANO_SAMPLE,sr=sample_rate,n_mfcc=n_mfcc)
Dataset_3 = extract_mfcc_pytorch(PIANO_SAMPLE,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels,mel_scale=mel_scale,n_mfcc=n_mfcc)
# plot_spectrogram(Dataset_2[0],title="MFCC")
# plot_spectrogram(Dataset_3[0],title="MFCC")
# plt.show()
#
# data = AudioDataset(TEST_SET_DIR)
# print(data.__getitem__(idx=0))
# print(data.__len__())
#
#
#
#
