// DOM Elements - File Upload
const audioFile = document.getElementById("audioFile");
const dropZone = document.getElementById("dropZone");
const analyzeButton = document.getElementById("analyzeButton");
const fileName = document.getElementById("fileName");
const loading = document.getElementById("loading");
const result = document.getElementById("result");
const resultIcon = document.getElementById("resultIcon");
const resultTitle = document.getElementById("resultTitle");
const confidence = document.getElementById("confidence");
const confidenceBar = document.getElementById("confidenceBar");
const duration = document.getElementById("duration");
const sampleRate = document.getElementById("sampleRate");
const resultMessage = document.getElementById("resultMessage");
const resetButton = document.getElementById("resetButton");

// DOM Elements - Live Mic Stream
const startMicBtn = document.getElementById("startMicBtn");
const stopMicBtn = document.getElementById("stopMicBtn");

let selectedFile = null;
let socket = null;
let audioContext = null;
let mediaStream = null;
let scriptProcessor = null;

// ==========================================
// FILE SELECTION & DRAG-AND-DROP
// ==========================================
audioFile.addEventListener("change", function () {
    if (this.files.length > 0) {
        selectedFile = this.files[0];
        showSelectedFile();
    }
});

dropZone.addEventListener("dragover", function (event) {
    event.preventDefault();
    dropZone.style.borderColor = "#4f46e5";
});

dropZone.addEventListener("dragleave", function () {
    dropZone.style.borderColor = "#cbd5e1";
});

dropZone.addEventListener("drop", function (event) {
    event.preventDefault();
    dropZone.style.borderColor = "#cbd5e1";
    if (event.dataTransfer.files.length > 0) {
        selectedFile = event.dataTransfer.files[0];
        showSelectedFile();
    }
});

function showSelectedFile() {
    if (!selectedFile) return;
    fileName.textContent = "Selected: " + selectedFile.name;
    analyzeButton.disabled = false;
}

// ==========================================
// FILE UPLOAD ANALYSIS (HTTP POST)
// ==========================================
analyzeButton.addEventListener("click", async function () {
    if (!selectedFile) {
        alert("Please select an audio file.");
        return;
    }

    analyzeButton.disabled = true;
    loading.classList.remove("hidden");
    result.classList.add("hidden");

    const formData = new FormData();
    formData.append("audio", selectedFile);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        loading.classList.add("hidden");

        if (!data.success) {
            alert("Error: " + data.error);
            analyzeButton.disabled = false;
            return;
        }

        displayResult(data);
    } catch (error) {
        console.error(error);
        loading.classList.add("hidden");
        alert("Could not connect to the server.");
        analyzeButton.disabled = false;
    }
});

// ==========================================
// LIVE MICROPHONE STREAMING (WEBSOCKET)
// ==========================================
if (startMicBtn && stopMicBtn) {
    startMicBtn.addEventListener("click", startLiveStream);
    stopMicBtn.addEventListener("click", stopLiveStream);
}

async function startLiveStream() {
    try {
        // Change address to your server IP for mobile (e.g., ws://192.168.x.x:8765)
        socket = new WebSocket("ws://192.168.0.106:8765");

        socket.onopen = async () => {
            startMicBtn.disabled = true;
            stopMicBtn.disabled = false;
            loading.classList.remove("hidden");

            mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 8000 });

            const source = audioContext.createMediaStreamSource(mediaStream);
            scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

            scriptProcessor.onaudioprocess = (event) => {
                if (socket.readyState === WebSocket.OPEN) {
                    const inputData = event.inputBuffer.getChannelData(0);
                    socket.send(inputData.buffer);
                }
            };

            source.connect(scriptProcessor);
            scriptProcessor.connect(audioContext.destination);
        };

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            loading.classList.add("hidden");
            
            // Map WebSocket payload to display formatting
            displayResult({
                confidence: data.ai_probability,
                duration: "Live",
                sample_rate: 8000,
                message: data.message,
                type: data.risk === "HIGH" || data.risk === "MEDIUM" ? "fake" : "human"
            });
        };

        socket.onclose = () => stopLiveStream();
    } catch (err) {
        console.error("Microphone Access Error:", err);
        alert("Could not access microphone.");
    }
}

function stopLiveStream() {
    if (scriptProcessor) scriptProcessor.disconnect();
    if (audioContext) audioContext.close();
    if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
    if (socket) socket.close();

    startMicBtn.disabled = false;
    stopMicBtn.disabled = true;
    loading.classList.add("hidden");
}

// ==========================================
// DISPLAY RESULT & RESET
// ==========================================
function displayResult(data) {
    result.classList.remove("hidden");

    confidence.textContent = data.confidence + "%";
    confidenceBar.style.setProperty("--confidence-width", data.confidence + "%");
    confidenceBar.style.width = data.confidence + "%";

    duration.textContent = data.duration + (typeof data.duration === "number" ? " sec" : "");
    sampleRate.textContent = data.sample_rate + " Hz";
    resultMessage.textContent = data.message;

    if (data.type === "fake") {
        resultIcon.textContent = "⚠️";
        resultTitle.textContent = "AI VOICE DETECTED";
    } else {
        resultIcon.textContent = "✅";
        resultTitle.textContent = "HUMAN VOICE";
    }
}

resetButton.addEventListener("click", function () {
    selectedFile = null;
    audioFile.value = "";
    fileName.textContent = "";
    analyzeButton.disabled = true;
    result.classList.add("hidden");
    confidenceBar.style.width = "0%";
    stopLiveStream();
});