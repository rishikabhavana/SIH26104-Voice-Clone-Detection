import asyncio
import json
import sounddevice as sd
import websockets

# Audio configuration matching the server
CHANNELS = 1
RATE = 8000
CHUNK = 4000  # Send 0.5 seconds of audio per chunk

async def audio_streamer():
    uri = "ws://localhost:8765"
    loop = asyncio.get_running_loop()
    audio_queue = asyncio.Queue()

    # Callback runs in a separate thread, safely passing data back to the async loop
    def callback(indata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        loop.call_soon_threadsafe(audio_queue.put_nowait, indata.copy().tobytes())

    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            
            # Send an initial connection event
            await websocket.send(json.dumps({"type": "connect", "status": "streaming"}))
            print("Streaming audio... Press Ctrl+C to stop.")
            
            # Open the microphone stream
            with sd.InputStream(
                samplerate=RATE, 
                channels=CHANNELS, 
                dtype='float32', 
                blocksize=CHUNK, 
                callback=callback
            ):
                while True:
                    # Fetch audio chunks from the queue and send to the WebSocket server
                    data = await audio_queue.get()
                    await websocket.send(data)
                    
    except websockets.exceptions.ConnectionClosed:
        print("Server closed the connection.")
    except Exception as e:
        print(f"Stream error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(audio_streamer())
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")