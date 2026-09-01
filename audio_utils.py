import io
import librosa
import numpy as np
import pandas as pd
import soundfile as sf


def load_audio(audio_path, sr=16000):
    """Loads an audio file from a file path into a float32 NumPy array."""
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    return y


def load_audio_bytes(audio_bytes, source_sr=16000):
    """Converts incoming raw audio bytes or web audio streams into a float32 NumPy array."""
    try:
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
        if len(audio) > 0:
            return audio
    except Exception:
        pass

    try:
        data, _ = sf.read(io.BytesIO(audio_bytes))
        if data.ndim > 1:
            data = np.mean(data, axis=1)  # Stereo to mono
        return data.astype(np.float32)
    except Exception as e:
        print(f"Error parsing audio bytes: {e}")
        return np.array([], dtype=np.float32)


def split_into_windows(y, sr=16000, window_sec=4.0, hop_sec=2.0):
    """Splits an audio array into overlapping windows for continuous segment analysis."""
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    if len(y) < window_samples:
        return [y]

    windows = []
    for start in range(0, len(y) - window_samples + 1, hop_samples):
        windows.append(y[start : start + window_samples])

    return windows


def extract_features(audio_input):
    """Extract 41 acoustic features from a file path or NumPy audio array."""
    sr = 16000

    if isinstance(audio_input, str):
        y, sr = librosa.load(audio_input, sr=sr, mono=True)
    elif isinstance(audio_input, np.ndarray):
        y = audio_input
        if len(y) == 0:
            return None
    else:
        raise ValueError(
            "Invalid audio input type. Expected file path or numpy array."
        )

    features = {}

    # MFCC Features (26 features)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"MFCC_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"MFCC_{i+1}_std"] = float(np.std(mfcc[i]))

    # Zero Crossing Rate (2 features)
    zcr = librosa.feature.zero_crossing_rate(y)
    features["ZCR_mean"] = float(np.mean(zcr))
    features["ZCR_std"] = float(np.std(zcr))

    # RMS Energy (2 features)
    rms = librosa.feature.rms(y=y)
    features["RMS_mean"] = float(np.mean(rms))
    features["RMS_std"] = float(np.std(rms))

    # Spectral Centroid (2 features)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["SpectralCentroid_mean"] = float(np.mean(spectral_centroid))
    features["SpectralCentroid_std"] = float(np.std(spectral_centroid))

    # Spectral Bandwidth (2 features)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["SpectralBandwidth_mean"] = float(np.mean(spectral_bandwidth))
    features["SpectralBandwidth_std"] = float(np.std(spectral_bandwidth))

    # Spectral Rolloff (2 features)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["SpectralRolloff_mean"] = float(np.mean(spectral_rolloff))
    features["SpectralRolloff_std"] = float(np.std(spectral_rolloff))

    # Pitch Features (4 features)
    try:
        pitch, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        pitch = pitch[~np.isnan(pitch)]

        if len(pitch) > 0:
            features["Pitch_mean"] = float(np.mean(pitch))
            features["Pitch_std"] = float(np.std(pitch))
            features["Pitch_min"] = float(np.min(pitch))
            features["Pitch_max"] = float(np.max(pitch))
        else:
            features["Pitch_mean"] = features["Pitch_std"] = features[
                "Pitch_min"
            ] = features["Pitch_max"] = 0.0
    except Exception:
        features["Pitch_mean"] = features["Pitch_std"] = features[
            "Pitch_min"
        ] = features["Pitch_max"] = 0.0

    # Audio Duration (1 feature)
    features["Duration"] = float(len(y) / sr)

    return features


def prepare_model_input(audio_input):
    """Extracts features and formats them into a pandas DataFrame ready for inference."""
    feats = extract_features(audio_input)
    if feats is None:
        return None
    return pd.DataFrame([feats])