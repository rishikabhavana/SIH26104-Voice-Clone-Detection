import os
import librosa
import numpy as np
import pandas as pd

# ==============================
# SETTINGS
# ==============================

DATASET_PATH = "audio_samples"
OUTPUT_FILE = "features.csv"

SAMPLE_RATE = 16000


# ==============================
# FEATURE EXTRACTION
# ==============================

def extract_features(file_path):

    try:
        # Load audio
        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        # Avoid empty audio
        if len(audio) == 0:
            return None

        features = {}

        # --------------------------------
        # 1. MFCC FEATURES
        # --------------------------------

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=13
        )

        for i in range(13):
            features[f"MFCC_{i+1}_mean"] = np.mean(mfcc[i])
            features[f"MFCC_{i+1}_std"] = np.std(mfcc[i])

        # --------------------------------
        # 2. ZERO CROSSING RATE
        # --------------------------------

        zcr = librosa.feature.zero_crossing_rate(audio)

        features["ZCR_mean"] = np.mean(zcr)
        features["ZCR_std"] = np.std(zcr)

        # --------------------------------
        # 3. RMS ENERGY
        # --------------------------------

        rms = librosa.feature.rms(y=audio)

        features["RMS_mean"] = np.mean(rms)
        features["RMS_std"] = np.std(rms)

        # --------------------------------
        # 4. SPECTRAL CENTROID
        # --------------------------------

        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )

        features["SpectralCentroid_mean"] = np.mean(
            spectral_centroid
        )

        features["SpectralCentroid_std"] = np.std(
            spectral_centroid
        )

        # --------------------------------
        # 5. SPECTRAL BANDWIDTH
        # --------------------------------

        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr
        )

        features["SpectralBandwidth_mean"] = np.mean(
            spectral_bandwidth
        )

        features["SpectralBandwidth_std"] = np.std(
            spectral_bandwidth
        )

        # --------------------------------
        # 6. SPECTRAL ROLLOFF
        # --------------------------------

        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=sr
        )

        features["SpectralRolloff_mean"] = np.mean(
            spectral_rolloff
        )

        features["SpectralRolloff_std"] = np.std(
            spectral_rolloff
        )

        # --------------------------------
        # 7. PITCH / F0
        # --------------------------------

        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )

        # Remove NaN values
        valid_f0 = f0[~np.isnan(f0)]

        if len(valid_f0) > 0:

            features["Pitch_mean"] = np.mean(valid_f0)
            features["Pitch_std"] = np.std(valid_f0)
            features["Pitch_min"] = np.min(valid_f0)
            features["Pitch_max"] = np.max(valid_f0)

        else:

            features["Pitch_mean"] = 0
            features["Pitch_std"] = 0
            features["Pitch_min"] = 0
            features["Pitch_max"] = 0

        # --------------------------------
        # 8. SPEECH DURATION
        # --------------------------------

        features["Duration"] = len(audio) / sr

        return features

    except Exception as e:

        print(f"ERROR processing {file_path}")
        print(e)

        return None


# ==============================
# CREATE DATASET
# ==============================

dataset = []

print("\n================================")
print("STARTING FEATURE EXTRACTION")
print("================================\n")


# --------------------------------
# REAL VOICES
# --------------------------------

real_folder = os.path.join(
    DATASET_PATH,
    "real"
)

print("Processing REAL voices...\n")

for filename in os.listdir(real_folder):

    if filename.lower().endswith(".wav"):

        file_path = os.path.join(
            real_folder,
            filename
        )

        print(f"REAL: {filename}")

        features = extract_features(file_path)

        if features is not None:

            features["label"] = 0
            features["filename"] = filename

            dataset.append(features)


# --------------------------------
# FAKE VOICES
# --------------------------------

fake_folder = os.path.join(
    DATASET_PATH,
    "fake"
)

print("\nProcessing FAKE voices...\n")

for filename in os.listdir(fake_folder):

    if filename.lower().endswith(".wav"):

        file_path = os.path.join(
            fake_folder,
            filename
        )

        print(f"FAKE: {filename}")

        features = extract_features(file_path)

        if features is not None:

            features["label"] = 1
            features["filename"] = filename

            dataset.append(features)


# ==============================
# SAVE DATASET
# ==============================

df = pd.DataFrame(dataset)

print("\n================================")
print("DATASET CREATED SUCCESSFULLY")
print("================================")

print(f"\nTotal samples: {len(df)}")

print(
    f"Real samples: {len(df[df['label'] == 0])}"
)

print(
    f"Fake samples: {len(df[df['label'] == 1])}"
)

print("\nFeature count:", len(df.columns) - 2)

print("\nDataset preview:")

print(df.head())


# Save CSV

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nFeatures saved to: {OUTPUT_FILE}")

print("\n================================")
print("DONE")
print("================================")