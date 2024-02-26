import os
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
from mfcc_extracter import extract_mfcc_LITE


sample_rate  = 44100
n_mfcc = 40
hop_length = 160
n_fft = 400
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
            raise ValueError(f"Directory not found: {directory}")

        for label in os.listdir(directory):
            label_dir = os.path.join(directory, label)
            if os.path.isdir(label_dir):
                self.label_to_index[label] = len(self.label_to_index)
                for file in os.listdir(label_dir):
                    if file.endswith('.wav'):
                        self.files.append(os.path.join(label_dir, file))
                        self.labels.append(self.label_to_index[label])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        # label = self.labels[idx]
        label = os.path.basename(os.path.dirname(filepath))


        waveform, _ = torchaudio.load(filepath)

        if sample_rate != self.sample_rate:
            resample_transform = T.Resample(orig_freq=waveform.sample_rate, new_freq=self.sample_rate)
            waveform = resample_transform(waveform)

        mfcc = extract_mfcc_LITE(waveform, self.sample_rate, self.n_mfcc)


        return mfcc, label[0]


TRAIN_DATA_DIR = r"C:\Users\y_mc\PycharmProjects\StromAI\Dataset\IRMAS-TrainingData"
#
train_data = AudioDataset(TRAIN_DATA_DIR)
train_loader = DataLoader(train_data,batch_size=32,shuffle=True)
print("TRAIN DATA SIZE :",len(train_loader))
print("DATA LOADER SIZE :",len(train_loader))

for mfcc,label in train_loader:
    # print(f"MFCC: shape: {mfcc.shape}:{mfcc}, Label: {label}")
    print(f"Shape of X [N, C, H, W]: {mfcc.shape}")
    print(f"Shape of y: {type(label)}")
    break
