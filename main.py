import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# Path to our audio file
audio_file = "audio_samples/real_voice.wav"


# Load audio
audio, sample_rate = librosa.load(audio_file, sr=None)

print("Audio loaded successfully!")
print("Sample rate:", sample_rate)
print("Audio duration:", len(audio) / sample_rate, "seconds")


# Create Mel Spectrogram
mel_spectrogram = librosa.feature.melspectrogram(
    y=audio,
    sr=sample_rate
)


# Convert to decibels
mel_spectrogram_db = librosa.power_to_db(
    mel_spectrogram,
    ref=np.max
)


# Display the spectrogram
plt.figure(figsize=(10, 5))

librosa.display.specshow(
    mel_spectrogram_db,
    sr=sample_rate,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram - Real Voice")
plt.tight_layout()
plt.show()