import os
import numpy as np
from hparams import hparams as hp
from dataset import raw_collate, discrete_collate, AudiobookDataset

import librosa
from model import build_model
import torch
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from train import load_checkpoint

use_cuda = torch.cuda.is_available()

def evaluate_model(model, data_loader, checkpoint_dir, limit_eval_to=5):
    """evaluate model and save generated wav and plot

    """
    test_path = data_loader.dataset.test_path
    test_files = os.listdir(test_path)
    counter = 0
    output_dir = os.path.join(checkpoint_dir,'eval')
    for f in test_files:
        if f[-7:] == "mel.npy":
            mel = np.load(os.path.join(test_path,f))
            wav = model.generate(mel)
            file_id = f[5:-8]
            # save wav
            wav_path = os.path.join(output_dir,"output_{}.wav".format(file_id))
            librosa.output.write_wav(wav_path, wav, sr=hp.sample_rate)
            # save wav plot
            fig_path = os.path.join(output_dir,"output_{}.png".format(file_id))
            fig = plt.plot(wav.reshape(-1))
            plt.savefig(fig_path)
            # clear fig to drawing to the same plot
            plt.clf()
            counter += 1
        # stop evaluation early via limit_eval_to
        if counter >= limit_eval_to:
            break

if __name__=="__main__":
    checkpoint_dir = "./data_dir/"
    checkpoint_path = hp.load_checkpoint
    data_root = hp.data_dir

    # make dirs, load dataloader and set up device
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir,'eval'), exist_ok=True)
    dataset = AudiobookDataset(data_root)
    if hp.input_type == 'raw':
        collate_fn = raw_collate
    elif hp.input_type == 'mixture':
        collate_fn = raw_collate
    elif hp.input_type in ['bits', 'mulaw']:
        collate_fn = discrete_collate
    else:
        raise ValueError("input_type:{} not supported".format(hp.input_type))
    data_loader = DataLoader(dataset, collate_fn=collate_fn, shuffle=True, num_workers=0, batch_size=hp.batch_size)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # build model, create optimizer
    model = build_model().to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=hp.initial_learning_rate, betas=(
        hp.adam_beta1, hp.adam_beta2),
        eps=hp.adam_eps, weight_decay=hp.weight_decay,
        amsgrad=hp.amsgrad)

    if hp.fix_learning_rate:
        print("using fixed learning rate of :{}".format(hp.fix_learning_rate))
    elif hp.lr_schedule_type == 'step':
        print("using exponential learning rate decay")
    elif hp.lr_schedule_type == 'noam':
        print("using noam learning rate decay")

    model = load_checkpoint(checkpoint_path, model, optimizer, False)
    print("loading model from checkpoint:{}".format(checkpoint_path))
    
    evaluate_model(model, data_loader, checkpoint_dir)


    
