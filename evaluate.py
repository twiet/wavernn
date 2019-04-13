import os, pickle
import numpy as np
from hparams import hparams as hp

import librosa
from model import build_model
import torch
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from audio import *
from preprocess import get_wav_mel
from train import load_checkpoint

import matplotlib
matplotlib.use('Agg')

use_cuda = torch.cuda.is_available()

def save_spectrogram_comparision(save_path, mel_gen, mel_true):
    plt.figure(figsize=(10, 10))
    plt.subplot(2,1,1)
    librosa.display.specshow(mel_true, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram (true)')
    plt.tight_layout()

    plt.subplot(2,1,2)
    librosa.display.specshow(mel_gen, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram (gen)')
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

def test_model(model, data_path, output_dir, limit_eval_to=5):
    """test model and save generated wav and plot

    """
    test_path = os.path.join(data_path, "test")
    with open(os.path.join(data_path,'test_set_ids.pkl'), 'rb') as f:
        test_files = pickle.load(f)

    eval_dir = os.path.join(output_dir, 'eval')
    for file_id in test_files:
        f = f"{file_id}_mel.npy"
        save_path = os.path.join(eval_dir, f"output_{file_id}.jpg")
        wav_path = os.path.join(eval_dir, f"output_{file_id}.wav")

        mel_true = np.load(os.path.join(test_path, f))
        wav = model.generate(mel_true)
        print(f"writing to {wav_path}")
        librosa.output.write_wav(wav_path, wav, sr=hp.sample_rate)

        wav, mel_gen = get_wav_mel(wav)
        save_spectrogram_comparision(save_path, mel_gen, mel_true)

if __name__=="__main__":
    output_dir = hp.output_dir
    checkpoint_path = hp.load_checkpoint
    data_root = hp.data_dir

    # make dirs, load dataloader and set up device
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'eval'), exist_ok=True)
   
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # build model, create optimizer
    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=hp.initial_learning_rate, betas=(
        hp.adam_beta1, hp.adam_beta2),
        eps=hp.adam_eps, weight_decay=hp.weight_decay,
        amsgrad=hp.amsgrad)

    model = load_checkpoint(checkpoint_path, model, optimizer, False)
    print("loading model from checkpoint:{}".format(checkpoint_path))

    test_model(model, data_root, output_dir)


    
