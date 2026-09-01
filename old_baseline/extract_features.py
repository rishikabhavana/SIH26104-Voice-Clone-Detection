import librosa
import numpy as np


# Path to audio
audio_file = "audio_samples/real_voice.wav"


# Load audio
audio, sample_rate = librosa.load(audio_file, sr=None)


print("Audio loaded successfully!")
print("Sample rate:", sample_rate)
print("Duration:", len(audio) / sample_rate, "seconds")


# Extract MFCC features
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=13
)


print("\nMFCC Shape:")
print(mfcc.shape)


print("\nMFCC Features:")
print(mfcc)


# Calculate average MFCC values
mfcc_mean = np.mean(mfcc, axis=1)


print("\nAverage MFCC Values:")
print(mfcc_mean)