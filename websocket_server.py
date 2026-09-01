import asyncio
import json
import websockets
import numpy as np
from audio_utils import load_audio_bytes
from detector import VoiceDetector  # Ensure this matches your class name in detector.py

class LiveCallDetector:
    def __init__(self, detector):
        self.detector = detector
        self.audio_buffer = np.array([], dtype=np.float32)
        self.sample_rate = 8000
        self.window_seconds = 4
        self.window_samples = self.sample_rate * self.window_seconds

    async def handle(self, websocket):
        print("Live call stream connected.")
        try:
            async for message in websocket:
                if isinstance(message, str):
                    await self.handle_event(message)
                else:
                    await self.handle_audio(message, websocket)
        except Exception as error:
            print("Stream error:", error)
        finally:
            print("Live call stream closed.")

    async def handle_event(self, message):
        try:
            event = json.loads(message)
            print("Stream event:", event)
        except json.JSONDecodeError:
            print("Non-JSON event:", message)

    async def handle_audio(self, data, websocket):
        audio = load_audio_bytes(data, source_sr=self.sample_rate)
        self.audio_buffer = np.concatenate([self.audio_buffer, audio])

        if len(self.audio_buffer) >= self.window_samples:
            window = self.audio_buffer[:self.window_samples]
            self.audio_buffer = self.audio_buffer[self.window_samples:]

            result = self.detector.predict_audio(window)
            print("LIVE DETECTION:", result)
            await websocket.send(json.dumps(result))

async def main():
    # Initialize your model from detector.py
    detector = VoiceDetector()
    live_detector = LiveCallDetector(detector)
    
    server = await websockets.serve(live_detector.handle, "0.0.0.0", 8765)
    print("WebSocket server running on ws://0.0.0.0:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())