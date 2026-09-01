import os
import uuid
import joblib
import librosa
import numpy as np

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename


# ==========================================
# FLASK CONFIGURATION
# ==========================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_PATH = "models/voice_detector.pkl"

ALLOWED_EXTENSIONS = {
    "wav",
    "mp3",
    "m4a",
    "ogg",
    "flac",
    "webm"
}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================
# LOAD MODEL
# ==========================================

print("Loading trained model...")

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    model = None
    print("ERROR loading model:")
    print(e)


# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(file_path):

    print("Extracting features...")

    audio, sample_rate = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    duration = librosa.get_duration(
        y=audio,
        sr=sample_rate
    )

    # --------------------------------------
    # MFCC
    # --------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    features = {}

    for i in range(13):

        features[f"MFCC_{i + 1}_mean"] = float(
            np.mean(mfcc[i])
        )

        features[f"MFCC_{i + 1}_std"] = float(
            np.std(mfcc[i])
        )

    # --------------------------------------
    # ZERO CROSSING RATE
    # --------------------------------------

    zcr = librosa.feature.zero_crossing_rate(audio)

    features["ZCR_mean"] = float(np.mean(zcr))
    features["ZCR_std"] = float(np.std(zcr))

    # --------------------------------------
    # RMS ENERGY
    # --------------------------------------

    rms = librosa.feature.rms(y=audio)

    features["RMS_mean"] = float(np.mean(rms))
    features["RMS_std"] = float(np.std(rms))

    # --------------------------------------
    # SPECTRAL CENTROID
    # --------------------------------------

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )

    features["SpectralCentroid_mean"] = float(
        np.mean(spectral_centroid)
    )

    features["SpectralCentroid_std"] = float(
        np.std(spectral_centroid)
    )

    # --------------------------------------
    # SPECTRAL BANDWIDTH
    # --------------------------------------

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate
    )

    features["SpectralBandwidth_mean"] = float(
        np.mean(spectral_bandwidth)
    )

    features["SpectralBandwidth_std"] = float(
        np.std(spectral_bandwidth)
    )

    # --------------------------------------
    # SPECTRAL ROLLOFF
    # --------------------------------------

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate
    )

    features["SpectralRolloff_mean"] = float(
        np.mean(spectral_rolloff)
    )

    features["SpectralRolloff_std"] = float(
        np.std(spectral_rolloff)
    )

    # --------------------------------------
    # PITCH
    # --------------------------------------

    try:

        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate
        )

        pitch = f0[~np.isnan(f0)]

        if len(pitch) > 0:

            features["Pitch_mean"] = float(
                np.mean(pitch)
            )

            features["Pitch_std"] = float(
                np.std(pitch)
            )

            features["Pitch_min"] = float(
                np.min(pitch)
            )

            features["Pitch_max"] = float(
                np.max(pitch)
            )

        else:

            features["Pitch_mean"] = 0.0
            features["Pitch_std"] = 0.0
            features["Pitch_min"] = 0.0
            features["Pitch_max"] = 0.0

    except Exception:

        features["Pitch_mean"] = 0.0
        features["Pitch_std"] = 0.0
        features["Pitch_min"] = 0.0
        features["Pitch_max"] = 0.0

    # --------------------------------------
    # DURATION
    # --------------------------------------

    features["Duration"] = float(duration)

    return features, sample_rate, duration


# ==========================================
# CHECK FILE
# ==========================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def index():

    return render_template("index.html")


# ==========================================
# AUDIO PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:

        return jsonify({
            "success": False,
            "error": "Model could not be loaded."
        }), 500

    if "audio" not in request.files:

        return jsonify({
            "success": False,
            "error": "No audio file received."
        }), 400

    audio_file = request.files["audio"]

    if audio_file.filename == "":

        return jsonify({
            "success": False,
            "error": "No file selected."
        }), 400

    if not allowed_file(audio_file.filename):

        return jsonify({
            "success": False,
            "error": "Unsupported audio format."
        }), 400

    # --------------------------------------
    # CREATE UNIQUE FILE NAME
    # --------------------------------------

    extension = audio_file.filename.rsplit(
        ".", 1
    )[1].lower()

    unique_name = (
        str(uuid.uuid4())
        + "."
        + extension
    )

    filename = secure_filename(unique_name)

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    audio_file.save(file_path)

    try:

        # ----------------------------------
        # FEATURE EXTRACTION
        # ----------------------------------

        features, sample_rate, duration = extract_features(
            file_path
        )

        # ----------------------------------
        # GET MODEL FEATURE NAMES
        # ----------------------------------

        if hasattr(model, "named_steps"):

            scaler = model.named_steps.get("scaler")

            if scaler is not None and hasattr(
                scaler,
                "feature_names_in_"
            ):

                expected_features = list(
                    scaler.feature_names_in_
                )

            else:

                expected_features = list(
                    features.keys()
                )

        else:

            expected_features = list(
                features.keys()
            )

        # ----------------------------------
        # CHECK MISSING FEATURES
        # ----------------------------------

        missing = [
            feature
            for feature in expected_features
            if feature not in features
        ]

        if missing:

            return jsonify({
                "success": False,
                "error": "Missing model features.",
                "missing_features": missing
            }), 500

        # ----------------------------------
        # CREATE FEATURE DICTIONARY
        # ----------------------------------

        ordered_features = {
            feature: features[feature]
            for feature in expected_features
        }

        # ----------------------------------
        # PREDICTION
        # ----------------------------------

        import pandas as pd

        features_df = pd.DataFrame(
            [ordered_features]
        )

        prediction = model.predict(
            features_df
        )[0]

        # ----------------------------------
        # CONFIDENCE
        # ----------------------------------

        confidence = 0.0

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                features_df
            )[0]

            confidence = float(
                np.max(probabilities) * 100
            )

        # ----------------------------------
        # RESULT
        # ----------------------------------

        if prediction == 1:

            result = "AI VOICE DETECTED"
            result_type = "fake"

        else:

            result = "HUMAN VOICE"
            result_type = "real"

        return jsonify({

            "success": True,

            "result": result,

            "type": result_type,

            "confidence": round(
                confidence,
                2
            ),

            "duration": round(
                duration,
                2
            ),

            "sample_rate": sample_rate,

            "message":
                "This audio is likely generated or modified using AI."
                if prediction == 1
                else
                "No strong evidence of AI-generated voice was detected."

        })

    except Exception as e:

        print("Prediction error:")
        print(e)

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500

    finally:

        # ----------------------------------
        # DELETE TEMP FILE
        # ----------------------------------

        try:

            if os.path.exists(file_path):

                os.remove(file_path)

        except Exception:

            pass


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/health")
def health():

    return jsonify({

        "status": "online",

        "model_loaded":
            model is not None

    })


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    print()
    print("================================")
    print("VOICE CLONE DETECTION SYSTEM")
    print("================================")
    print("Starting Flask server...")
    print("Open: http://127.0.0.1:5000")
    print("================================")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )