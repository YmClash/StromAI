from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy
import torchaudio
import os
from mfcc_extracter import extract_mfcc_LITE
import pandas as pd

DATASET_DIR = r"C:\Users\y_mc\PycharmProjects\StromAI\Dataset\Test_Set"

RESULTS_DIRECTORY = DATASET_DIR + "/MFCC_RESULTS"

sample_rate = 44100  # Fréquence d'échantillonnage
n_mfcc = 40  # Nombre de coefficients MFCC
hop_length = 160  # Longueur du saut
n_fft = 400  # Taille de la fenêtre FFT
n_mels = 40  # Nombre de bandes Mel
mel_scale = 'htk'  # Échelle Mel

print(f"Sample rate : {sample_rate}\nn_mfcc: {n_mfcc}\n")

print("Debut de l'extraction...")
def extract_to_csv(filepath,csv_file):

    waveform,_ = torchaudio.load(filepath)

    mfcc = extract_mfcc_LITE(waveform,sample_rate,n_mfcc)

    filename = os.path.basename(filepath)
    label = os.path.basename(os.path.dirname(filepath))

    with open(csv_file, 'a') as file:
        file.write(f"{filename},{label},{mfcc.tolist()}\n")


csv_file = r"C:\Users\y_mc\PycharmProjects\StromAI\Dataset\IRMAS-TrainingData\MFCC_RESULTS\mfcc.csv"

with open(csv_file,'w') as file :
    file.write("filename,label,mfcc\n")

for filepath in os.listdir(DATASET_DIR):
    if filepath.endswith('.wav'):
        extract_to_csv(os.path.join(DATASET_DIR,filepath),csv_file)


print()
print("Done.....")

data_frame = pd.read_csv(csv_file)

print(data_frame.head())






# Extraction  a

"""
if not os.path.exists(RESULTS_DIRECTORY) :
    os.makedirs(RESULTS_DIRECTORY)
    print("Creatin New Folder ......")

print("Start Extraction")

for filename in os.listdir(DATA_DIRECTORY) :
    if filename.endswith('.wav') :
        rate, sig = wav.read(DATA_DIRECTORY + "/" + filename)

        mfcc_feat = mfcc(sig, rate)

        fil_bank_feat = logfbank(sig, rate)

        outputFile = RESULTS_DIRECTORY + "/" + os.path.splitext(filename)[0] + ".csv"
        file = open(outputFile, "w+")
        numpy.savetxt(file, fil_bank_feat, delimiter=",")
        file.close()

    print("Done..............")
"""