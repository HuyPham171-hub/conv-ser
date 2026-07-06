import librosa
import numpy as np
import random

class AudioAugmentor:
    def __init__(self, sr=16000):
        self.sr = sr

    def add_awgn(self, samples, snr_db=20):
        rms = np.sqrt(np.mean(samples**2))
        noise_rms = rms / (10**(snr_db / 20))
        noise = np.random.normal(0, noise_rms, samples.shape)
        return samples + noise

    def pitch_shift(self, samples, steps):
        return librosa.effects.pitch_shift(samples, sr=self.sr, n_steps=steps)

    def time_stretch(self, samples, rate):
        return librosa.effects.time_stretch(samples, rate=rate)

    def freq_mask(self, mel, freq_mask_param=20):
        num_mel = mel.shape[0]
        f = np.random.randint(0, freq_mask_param)
        f0 = random.randint(0, num_mel - f)
        mel[f0:f0+f, :] = 0
        return mel

    def time_mask(self, mel, time_mask_param=30):
        num_frames = mel.shape[1]
        t = np.random.randint(0, time_mask_param)
        t0 = random.randint(0, num_frames - t)
        mel[:, t0:t0+t] = 0
        return mel

    def spec_augment(self, samples, n_mels=128):
        mel = librosa.feature.melspectrogram(samples, sr=self.sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = self.freq_mask(mel_db)
        mel_db = self.time_mask(mel_db)
        return mel_db

    def augment_audio(self, samples):
        """For SER model or balancing"""
        ops = [
            lambda x: self.add_awgn(x, snr_db=random.choice([10,15,20])),
            lambda x: self.pitch_shift(x, steps=random.uniform(-2,2)),
            lambda x: self.time_stretch(x, rate=random.uniform(0.9,1.1)),
        ]
        op = random.choice(ops)
        return op(samples)

    def augment_spec(self, samples):
        """For CNN model using spectrogram"""
        return self.spec_augment(samples)
