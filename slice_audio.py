import os, random, librosa, musdb
from hparams import hparams as hp
from audio import *
from tqdm import tqdm

def slice_wav(wav, duration=10, max_slices=5):
    window = hp.sample_rate * duration
    stride = int(window * 0.8)
    slices = get_wav_slices(wav, window, stride)
    out = [wav[j:k] for j,k in slices if sum(wav[j:k]) != 0]
    if len(out) > max_slices:
        return out
    return random.sample(out, max_slices)

if __name__=="__main__":
    wav_dir = hp.dataset
    sliced_dir = os.path.join(wav_dir, "sliced")
    os.makedirs(sliced_dir, exist_ok=True)
    is_musdb = True
    if is_musdb:
        sliced_dir = os.path.join(hp.musdb18_path, "sliced")
        os.makedirs(sliced_dir, exist_ok=True)

        samples = 30
        mus = musdb.DB(root_dir=hp.musdb18_path)
        tracks = random.sample(mus.load_mus_tracks(subsets=['test']), samples)
        for track in tqdm(tracks):
            mixture = track.audio
            vocal = track.targets['vocals'].audio
            sample_rate = track.rate
            mix_wav = librosa.to_mono(mixture.T)
            vox_wav = librosa.to_mono(vocal.T)
            if sample_rate != hp.sample_rate:
                mix_wav = librosa.resample(mix_wav, sample_rate, hp.sample_rate)
                vox_wav = librosa.resample(vox_wav, sample_rate, hp.sample_rate)

            # focus only on vocals for now
            # wav_slices = slice_wav(mix_wav)
            # for i, wav in enumerate(wav_slices):
            #     path = os.path.join(sliced_dir, f"{track.name}_{i:03d}_mix.wav")
            #     save_wav(wav, path)

            wav_slices = slice_wav(vox_wav)
            for i, wav in enumerate(wav_slices):
                path = os.path.join(sliced_dir, "{}_{03d}_vox.wav".format(track.name, i))
                save_wav(wav, path)
                