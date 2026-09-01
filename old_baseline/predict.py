import os
import joblib
import librosa
import numpy as np
import pandas as pd


# ============================================
# CONFIGURATION
# ============================================

AUDIO_FILE = "audio_samples/real/real_voice.wav"
MODEL_FILE = "models/voice_detector.pkl"


# ============================================
# FEATURE EXTRACTION
# EXACTLY MATCHES prepare_dataset.py
# ============================================

def extract_features(audio_file):

    # Load audio
    audio, sr = librosa.load(
        audio_file,
        sr=None
    )

    duration = librosa.get_duration(
        y=audio,
        sr=sr
    )

    features = {}

    # ========================================
    # 1. MFCC
    # ========================================

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13
    )

    for i in range(13):

        features[f"MFCC_{i+1}_mean"] = np.mean(
            mfcc[i]
        )

        features[f"MFCC_{i+1}_std"] = np.std(
            mfcc[i]
        )


    # ========================================
    # 2. ZERO CROSSING RATE
    # ========================================

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )

    features["ZCR_mean"] = np.mean(zcr)
    features["ZCR_std"] = np.std(zcr)


    # ========================================
    # 3. RMS ENERGY
    # ========================================

    rms = librosa.feature.rms(
        y=audio
    )

    features["RMS_mean"] = np.mean(rms)
    features["RMS_std"] = np.std(rms)


    # ========================================
    # 4. SPECTRAL CENTROID
    # ========================================

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


    # ========================================
    # 5. SPECTRAL BANDWIDTH
    # ========================================

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


    # ========================================
    # 6. SPECTRAL ROLLOFF
    # ========================================

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


    # ========================================
    # 7. PITCH
    # ========================================

    pitch, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )

    # Remove NaN values
    valid_pitch = pitch[
        ~np.isnan(pitch)
    ]

    if len(valid_pitch) > 0:

        features["Pitch_mean"] = np.mean(
            valid_pitch
        )

        features["Pitch_std"] = np.std(
            valid_pitch
        )

        features["Pitch_min"] = np.min(
            valid_pitch
        )

        features["Pitch_max"] = np.max(
            valid_pitch
        )

    else:

        features["Pitch_mean"] = 0
        features["Pitch_std"] = 0
        features["Pitch_min"] = 0
        features["Pitch_max"] = 0


    # ========================================
    # 8. AUDIO DURATION
    # ========================================

    features["Duration"] = duration


    return features, sr, duration


# ============================================
# START PROGRAM
# ============================================

print()
print("================================")
print("VOICE CLONE DETECTION")
print("================================")


# ============================================
# LOAD MODEL
# ============================================

print()
print("Loading model...")

model = joblib.load(
    MODEL_FILE
)

print("Model loaded successfully!")


# ============================================
# CHECK AUDIO FILE
# ============================================

if not os.path.exists(AUDIO_FILE):

    print()
    print("ERROR: Audio file not found!")
    print("File:", AUDIO_FILE)

    exit()


# ============================================
# ANALYZE AUDIO
# ============================================

print()
print("Analyzing audio...")
print("File:", AUDIO_FILE)


features, sample_rate, duration = extract_features(
    AUDIO_FILE
)


print("Sample rate:", sample_rate)
print(
    "Duration:",
    round(duration, 2),
    "seconds"
)


# ============================================
# CREATE DATAFRAME
# ============================================

features_df = pd.DataFrame(
    [features]
)


print()
print("================================")
print("FEATURE EXTRACTION")
print("================================")

print(
    "Features extracted:",
    len(features_df.columns)
)


# ============================================
# GET FEATURES EXPECTED BY MODEL
# ============================================

if hasattr(model, "named_steps"):

    scaler = model.named_steps.get(
        "scaler"
    )

    if scaler is not None and hasattr(
        scaler,
        "feature_names_in_"
    ):

        expected_features = list(
            scaler.feature_names_in_
        )

        print(
            "Model expects:",
            len(expected_features),
            "features"
        )

        # Check missing features
        missing = set(
            expected_features
        ) - set(
            features_df.columns
        )

        if missing:

            print()
            print("ERROR: Missing features:")
            print(missing)

            exit()

        # IMPORTANT:
        # Put features in EXACT training order

        features_df = features_df[
            expected_features
        ]


# ============================================
# FINAL FEATURE CHECK
# ============================================

if len(features_df.columns) != 41:

    print()
    print(
        "ERROR: Feature count mismatch!"
    )

    print(
        "Expected: 41"
    )

    print(
        "Received:",
        len(features_df.columns)
    )

    exit()


print()
print(
    "Feature count verified: 41"
)


# ============================================
# PREDICTION
# ============================================

print()
print("Running prediction...")


prediction = model.predict(
    features_df
)[0]


# ============================================
# CONFIDENCE
# ============================================

try:

    probabilities = model.predict_proba(
        features_df
    )[0]

    confidence = (
        np.max(probabilities) * 100
    )

except Exception:

    confidence = None


# ============================================
# RESULT
# ============================================

print()
print("================================")
print("DETECTION RESULT")
print("================================")


if prediction == 1:

    print()
    print("RESULT: FAKE / AI-GENERATED VOICE")

else:

    print()
    print("RESULT: REAL / HUMAN VOICE")


if confidence is not None:

    print(
        "Confidence:",
        round(confidence, 2),
        "%"
    )


print()
print("================================")
print("ANALYSIS COMPLETE")
print("================================")