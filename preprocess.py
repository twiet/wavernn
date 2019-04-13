import numpy as np
import math, pickle, os, sys
from audio import *
from hparams import hparams as hp
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display
import librosa
import musdb

import matplotlib
matplotlib.use('Agg')

def slice_wav(wav, duration=5, max_slices=5):
    window = hp.sample_rate * duration
    stride = int(window * 0.8)
    slices = get_wav_slices(wav, window, stride)
    return [wav[j:k] for j,k in slices if sum(wav[j:k]) != 0][:max_slices]

def save_spectrogram(save_path, mel):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_wav_mel(wav):
    """Given path to .wav file, get the quantized wav and mel spectrogram as numpy vectors

    """
    mel = melspectrogram(wav)
    if hp.input_type == 'raw':
        return wav.astype(np.float32), mel
    elif hp.input_type == 'mulaw':
        quant = mulaw_quantize(wav, hp.mulaw_quantize_channels)
        return quant.astype(np.int), mel
    elif hp.input_type == 'bits':
        quant = quantize(wav)
        return quant.astype(np.int), mel
    else:
        raise ValueError("hp.input_type {} not recognized".format(hp.input_type))

def preprocess(wav_files, wav_dir, mel_path, wav_path, mode = "train"):
    dataset_ids = []
    for _, wav_file in enumerate(tqdm(wav_files)):
        file_path = os.path.join(wav_dir, wav_file)
        wav, _ = load_wav(file_path)
        wav, mel = get_wav_mel(wav)
        file_id = f"{wav_file[:-4]}"
        save_spectrogram(os.path.join(mel_path, f"{file_id}.jpg"), mel)
        save_wav(wav, os.path.join(wav_path, f"{file_id}.wav"))

        np.save(os.path.join(mel_path, f"{file_id}_mel.npy"), mel)
        np.save(os.path.join(wav_path, f"{file_id}_wav.npy"), wav)
        dataset_ids.append(file_id)
    return dataset_ids

def process_data(wav_dir, output_path, mel_path, wav_path, mode="train"):
    """
    given wav directory and output directory, process wav files and save quantized wav and mel
    spectrogram to output directory
    """
    wav_files = [file for file in os.listdir(wav_dir) if file[-4:] == '.wav']
    assert len(wav_files) != 0 or wav_files[0][-4:] == '.wav', "no wav files found!"

    test_wav_files = wav_files
    dataset_ids = []
    if mode == "train":
        test_wav_files = wav_files[:hp.test_split]
        wav_files = wav_files[hp.test_split:]
        dataset_ids = preprocess(wav_files, wav_dir, mel_path, wav_path, mode=mode)

    with open(os.path.join(output_path, 'training_set_ids.pkl'), 'wb') as f:
        pickle.dump(dataset_ids, f)

    test_path = os.path.join(output_path, 'test')
    os.makedirs(test_path, exist_ok=True)
    dataset_ids = preprocess(test_wav_files, wav_dir, test_path, test_path, mode=mode)
    with open(os.path.join(output_path, 'test_set_ids.pkl'), 'wb') as f:
        pickle.dump(dataset_ids, f)
    print("\npreprocessing done, total processed wav files:{}.\nProcessed files are located in:{}".format(len(wav_files), os.path.abspath(output_path)))

def preprocess_test(mode="test"):
    wav_dir = hp.dataset
    output_dir = hp.data_dir

    # create paths
    output_path = os.path.join(output_dir, "")
    mel_path = os.path.join(output_dir, "mel")
    wav_path = os.path.join(output_dir, "wav")

    # create dirs
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(mel_path, exist_ok=True)
    os.makedirs(wav_path, exist_ok=True)

    # process data
    process_data(wav_dir, output_path, mel_path, wav_path, mode=mode)

if __name__=="__main__":
    preprocess_test("train")
