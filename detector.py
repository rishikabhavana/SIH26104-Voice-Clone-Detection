import os
import numpy as np
import torch

from audio_utils import load_audio, prepare_model_input, split_into_windows

class VoiceDeepfakeDetector:
    def __init__(self, model=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model

        print("=" * 50)
        print("VOICE DEEPFAKE DETECTOR")
        print("=" * 50)
        print("Device:", self.device)

    def predict_window(self, audio):
        """
        Run the neural detector on one audio window.
        IMPORTANT: Replace the model-loading implementation with the exact 
        RawNet2 checkpoint architecture used by your downloaded checkpoint.
        """
        audio = prepare_model_input(audio)
        tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            logits = output["logits"] if isinstance(output, dict) else output
            probabilities = torch.softmax(logits, dim=-1)

        return {
            "ai_probability": float(probabilities[0][0].item()),
            "human_probability": float(probabilities[0][1].item())
        }

    def predict_file(self, file_path):
        audio, sr = load_audio(file_path)
        return self._process_windows(split_into_windows(audio))

    def predict_audio(self, audio):
        return self._process_windows(split_into_windows(audio))

    def _process_windows(self, windows):
        results = []
        for index, window in enumerate(windows):
            result = self.predict_window(window)
            result["window"] = index + 1
            results.append(result)

        if not results:
            return self._unknown_result()

        ai_scores = [r["ai_probability"] for r in results]
        final_ai = (0.7 * float(np.mean(ai_scores))) + (0.3 * float(np.max(ai_scores)))
        
        return self._make_decision(final_ai, results)

    def _make_decision(self, ai_probability, windows):
        ai_percentage = round(ai_probability * 100, 2)
        human_percentage = round((1 - ai_probability) * 100, 2)

        if ai_probability >= 0.80:
            label, risk, color = "AI-GENERATED VOICE", "HIGH", "danger"
            message = "Speech characteristics consistent with AI-generated or manipulated audio were detected."
        elif ai_probability >= 0.60:
            label, risk, color = "SUSPICIOUS VOICE", "MEDIUM", "warning"
            message = "The audio contains characteristics that may indicate synthetic speech."
        else:
            label, risk, color = "HUMAN VOICE", "LOW", "safe"
            message = "No strong evidence of AI-generated speech was detected."

        return {
            "label": label,
            "risk": risk,
            "color": color,
            "ai_probability": ai_percentage,
            "human_probability": human_percentage,
            "message": message,
            "windows": windows
        }

    def _unknown_result(self):
        return {
            "label": "UNKNOWN",
            "risk": "UNKNOWN",
            "color": "warning",
            "ai_probability": 0,
            "human_probability": 0,
            "message": "Unable to analyze the audio."
        }