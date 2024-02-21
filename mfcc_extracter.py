import librosa
import torch
import torchaudio
from torchaudio.transforms import MFCC
import os

def extract_mfcc(dir_path, sr, n_mfcc):
    mfcc_list = []
    if isinstance(dir_path, (list, tuple)):
        for file in dir_path:
            audio, sr = librosa.load(file, sr=sr)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            # #####
            print(f'MFFC SHAPE : {mfcc.shape}')
            mfcc_list.append(mfcc)
    elif isinstance(dir_path, str):
        for filename in os.listdir(dir_path):
            filepath = os.path.join(dir_path, filename)
            audio, sr = librosa.load(filepath, sr=sr)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            print(f'MFFC SHAPE : {mfcc.shape}')

            mfcc_list.append(mfcc)
    else:
        raise ValueError("Invalid file_path format. Must be a string directory path or a list of file paths")
    print("EXTRACT AUDIO FOLDER FILE ")
    print(mfcc_list)
    print(len(mfcc_list))
    return mfcc_list

def extract_mfcc_LITE(file_path, sr, n_mfcc):
    # audio, sr = librosa.load(file_path, sr=sr)
    audio = file_path.numpy().squeeze()

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    print("MFCC SINGLE")
    print(mfcc)
    print(mfcc.shape)
    return mfcc

def extract_mfcc_pytorch(file_path, n_fft, hop_length, n_mels, mel_scale, n_mfcc):
    audio, sr = torchaudio.load(file_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    mfcc_transform = MFCC(sample_rate=sr, n_mfcc=n_mfcc, melkwargs={"n_fft": n_fft, "n_mels": n_mels, "hop_length": hop_length, "mel_scale": mel_scale})
    mfcc = mfcc_transform(audio)
    print("Extract with MFCC PYTORCH")
    print(mfcc)
    print(mfcc.shape)
    return mfcc
