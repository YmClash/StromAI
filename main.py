import librosa

import torch
import torchaudio
from torchaudio.transforms import MFCC

file_path = "chemin_vers_votre_fichier_audio"
n_mfcc = 40
n_fft = 400
hop_length = 160
n_mels = 40
mel_scale = 'htk'

# waveform, sample_rate = torchaudio.load(file_path)
# if waveform.shape[0] > 1:
#     waveform = torch.mean(waveform, dim=0, keepdim=True)
# mfcc_transform = MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs={"n_fft": n_fft, "n_mels": n_mels, "hop_length": hop_length, "mel_scale": mel_scale})
# mfcc = mfcc_transform(waveform)


DATA = r"C:\Users\y_mc\PycharmProjects\StromAI\Dataset\Test_Set\[pia][cla]1283__1.wav"


audio, sr = librosa.load(DATA)
mfcc_1= librosa.feature.mfcc(y=audio,sr=sr,)

waveform ,sample_rate  = torchaudio.load(DATA)
print(f"Wave shape :{waveform.shape[0]}")
if waveform.shape[0] > 1 :
    waveform = torch.mean(waveform,dim=0 ,keepdim=True)
    print("wave shape",waveform.shape)
    print("wave dim :",waveform.dim())
    print()

mfcc_transform = MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs={"n_fft": n_fft, "n_mels": n_mels, "hop_length": hop_length, "mel_scale": mel_scale})
mfcc_2 = mfcc_transform(waveform)
print(mfcc_2)
print(mfcc_2.shape)
print()
# mfcc_2 = to22rchau
# print(mfcc_1)
print(f"MFCC 1 shape :{mfcc_1.shape} ")
print(f"MFCC 2 shape :{mfcc_2.shape} ")
print("mfcc1 :", mfcc_1.ndim)
print("mfcc 2 : ",mfcc_2.ndim)



# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')






# See PyCharm help at https://www.jetbrains.com/help/pycharm/
