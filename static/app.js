/**
 * SignBridge AI - app.js
 * Two-way sign language communication system
 * Browser handles: Camera, MediaPipe hand landmarks, Speech API
 * Backend handles: Landmark classification via RandomForest
 */

import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

// ── Config ──────────────────────────────────────────────────────────
const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
const WS_URL = `${wsProtocol}//${location.host}/ws/predict`;
const CONFIDENCE_THRESHOLD = 0.65;
const STABILITY_FRAMES = 15;       // frames same prediction needed before output
const COOLDOWN_MS = 2500;          // ms before same word can be spoken again

// ── Vocabulary & Sign Visuals ────────────────────────────────────────
const VOCABULARY = {
  "hello":     { label: "Hello",     file: "hello" },
  "yes":       { label: "Yes",       file: "yes" },
  "no":        { label: "No",        file: "no" },
  "help":      { label: "Help",      file: "help" },
  "thank you": { label: "Thank You", file: "thank_you" },
  "thank":     { label: "Thank You", file: "thank_you" },
  "please":    { label: "Please",    file: "please" },
  "sorry":     { label: "Sorry",     file: "sorry" },
  "welcome":   { label: "Welcome",   file: "welcome" },
  "stop":      { label: "Stop",      file: "stop" },
  "good":      { label: "Good",      file: "good" },
  "bad":       { label: "Bad",       file: "bad" },
  "wait":      { label: "Wait",      file: "wait" },
};

// ── DOM refs ─────────────────────────────────────────────────────────
const webcamEl       = document.getElementById("webcam");
const overlayCanvas  = document.getElementById("overlayCanvas");
const ctx            = overlayCanvas.getContext("2d");
const cameraOverlay  = document.getElementById("cameraOverlay");
const startCamBtn    = document.getElementById("startCam");
const stopCamBtn     = document.getElementById("stopCam");
const detectionStatus= document.getElementById("detectionStatus");
const detectedSign   = document.getElementById("detectedSign");
const confidenceBar  = document.getElementById("confidenceBar");
const confidenceText = document.getElementById("confidenceText");
const textOutput     = document.getElementById("textOutput");
const speakStatus    = document.getElementById("speakStatus");
const clearBtn       = document.getElementById("clearBtn");
const modelStatus    = document.getElementById("modelStatus");
const wsStatusEl     = document.getElementById("wsStatus");

const micBtn         = document.getElementById("micBtn");
const micStatus      = document.getElementById("micStatus");
const transcriptEl   = document.getElementById("transcript");
const textInput      = document.getElementById("textInput");
const translateBtn   = document.getElementById("translateBtn");
const recognizedKw   = document.getElementById("recognizedKeyword");
const signVisualArea = document.getElementById("signVisualArea");
const chips          = document.querySelectorAll(".chip");

// ── State ────────────────────────────────────────────────────────────
let stream = null;
let handLandmarker = null;
let animFrameId = null;
let ws = null;
let wsReady = false;

let predictionBuffer = [];
let lastSpokenSign = null;
let lastSpokenTime = 0;
let isSpeaking = false;

let recognition = null;
let recognizing = false;

// ── Init ──────────────────────────────────────────────────────────────
async function init() {
  await checkHealth();
  await initMediaPipe();
  connectWebSocket();
  setupEventListeners();
}

// ── Health check ──────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.model_loaded) {
      setBadge(modelStatus, "Model: Ready", "success");
    } else {
      setBadge(modelStatus, "Model: Not Trained", "warning");
    }
  } catch {
    setBadge(modelStatus, "Model: Error", "error");
  }
}

// ── MediaPipe ─────────────────────────────────────────────────────────
async function initMediaPipe() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 1,
    });
    console.log("[SignBridge] MediaPipe HandLandmarker ready");
  } catch (err) {
    console.error("[SignBridge] MediaPipe init error:", err);
    setStatus(detectionStatus, "MediaPipe Error", "error");
  }
}

// ── WebSocket ─────────────────────────────────────────────────────────
function connectWebSocket() {
  try {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      wsReady = true;
      setBadge(wsStatusEl, "WS: Connected", "success");
      console.log("[WS] Connected");
    };

    ws.onclose = () => {
      wsReady = false;
      setBadge(wsStatusEl, "WS: Disconnected", "error");
      console.log("[WS] Disconnected, retrying in 3s...");
      setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (e) => {
      console.warn("[WS] Error:", e);
      setBadge(wsStatusEl, "WS: Error", "error");
    };

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        if (data.type === "prediction") {
          handlePrediction(data);
        } else if (data.type === "error") {
          console.warn("[WS] Server error:", data.message);
        }
      } catch (e) {
        console.warn("[WS] Parse error:", e);
      }
    };
  } catch (err) {
    console.error("[WS] Cannot connect:", err);
  }
}

// ── Landmark normalization (mirrors Python backend) ────────────────────
function normalizeLandmarks(landmarks) {
  // landmarks: array of {x, y, z} objects, 21 items
  const arr = landmarks.map(lm => [lm.x, lm.y, lm.z]);

  // Subtract wrist
  const wrist = arr[0].slice();
  for (let i = 0; i < arr.length; i++) {
    arr[i][0] -= wrist[0];
    arr[i][1] -= wrist[1];
    arr[i][2] -= wrist[2];
  }

  // Find max distance
  let maxDist = 0;
  for (const pt of arr) {
    const d = Math.sqrt(pt[0]*pt[0] + pt[1]*pt[1] + pt[2]*pt[2]);
    if (d > maxDist) maxDist = d;
  }

  // Scale
  const flat = [];
  for (const pt of arr) {
    flat.push(maxDist > 0 ? pt[0]/maxDist : pt[0]);
    flat.push(maxDist > 0 ? pt[1]/maxDist : pt[1]);
    flat.push(maxDist > 0 ? pt[2]/maxDist : pt[2]);
  }
  return flat; // 63 values
}

// ── Camera ───────────────────────────────────────────────────────────
async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showCameraError("Camera API not supported in this browser.");
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    webcamEl.srcObject = stream;
    await new Promise((r) => (webcamEl.onloadedmetadata = r));
    webcamEl.play();

    cameraOverlay.classList.add("hidden");
    startCamBtn.disabled = true;
    stopCamBtn.disabled = false;
    setStatus(detectionStatus, "Camera Active", "success");

    startDetectionLoop();
  } catch (err) {
    if (err.name === "NotAllowedError") {
      showCameraError("Camera permission denied. Please allow camera access.");
    } else if (err.name === "NotFoundError") {
      showCameraError("No camera found on this device.");
    } else {
      showCameraError(`Camera error: ${err.message}`);
    }
  }
}

function stopCamera() {
  if (animFrameId) cancelAnimationFrame(animFrameId);
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  webcamEl.srcObject = null;
  cameraOverlay.classList.remove("hidden");
  startCamBtn.disabled = false;
  stopCamBtn.disabled = true;
  setStatus(detectionStatus, "Idle", "idle");
  clearDetection();
}

function showCameraError(msg) {
  cameraOverlay.classList.remove("hidden");
  cameraOverlay.innerHTML = `<span>&#128247;</span><p style="color:#f85149">${msg}</p>`;
}

// ── Detection loop ────────────────────────────────────────────────────
let lastVideoTime = -1;

function startDetectionLoop() {
  function detect() {
    if (!stream) return;

    overlayCanvas.width  = webcamEl.videoWidth  || 640;
    overlayCanvas.height = webcamEl.videoHeight || 480;
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    if (!handLandmarker) {
      animFrameId = requestAnimationFrame(detect);
      return;
    }

    const now = performance.now();
    if (webcamEl.currentTime !== lastVideoTime) {
      lastVideoTime = webcamEl.currentTime;

      let result;
      try {
        result = handLandmarker.detectForVideo(webcamEl, now);
      } catch (e) {
        animFrameId = requestAnimationFrame(detect);
        return;
      }

      if (result && result.landmarks && result.landmarks.length > 0) {
        const handLandmarks = result.landmarks[0];

        // Draw on canvas
        drawLandmarks(handLandmarks, overlayCanvas.width, overlayCanvas.height);

        setStatus(detectionStatus, "Hand Detected", "active");

        // Send to backend for prediction
        if (wsReady && ws && ws.readyState === WebSocket.OPEN) {
          const flat = normalizeLandmarks(handLandmarks);
          ws.send(JSON.stringify({ landmarks: flat }));
        }
      } else {
        setStatus(detectionStatus, "No Hand", "idle");
        predictionBuffer = [];
      }
    }

    animFrameId = requestAnimationFrame(detect);
  }
  animFrameId = requestAnimationFrame(detect);
}

// ── Draw landmarks ────────────────────────────────────────────────────
const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],       // thumb
  [0,5],[5,6],[6,7],[7,8],       // index
  [0,9],[9,10],[10,11],[11,12],  // middle
  [0,13],[13,14],[14,15],[15,16],// ring
  [0,17],[17,18],[18,19],[19,20],// pinky
  [5,9],[9,13],[13,17],          // palm
];

function drawLandmarks(landmarks, w, h) {
  ctx.strokeStyle = "#58a6ff";
  ctx.lineWidth = 2;

  // Connections
  for (const [a, b] of CONNECTIONS) {
    const p1 = landmarks[a], p2 = landmarks[b];
    ctx.beginPath();
    ctx.moveTo(p1.x * w, p1.y * h);
    ctx.lineTo(p2.x * w, p2.y * h);
    ctx.stroke();
  }

  // Points
  for (const lm of landmarks) {
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#7c3aed";
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

// ── Prediction handling with stabilization ────────────────────────────
function handlePrediction(data) {
  const { sign, confidence } = data;

  if (confidence < CONFIDENCE_THRESHOLD) {
    return;
  }

  predictionBuffer.push(sign);
  if (predictionBuffer.length > STABILITY_FRAMES) {
    predictionBuffer.shift();
  }

  if (predictionBuffer.length < STABILITY_FRAMES) return;

  // Check if all frames agree
  const counts = {};
  for (const s of predictionBuffer) {
    counts[s] = (counts[s] || 0) + 1;
  }
  const topSign = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  if (topSign[1] < Math.floor(STABILITY_FRAMES * 0.8)) return;

  const stableSign = topSign[0];

  // Update UI
  detectedSign.textContent = stableSign;
  confidenceBar.style.width = `${Math.round(confidence * 100)}%`;
  confidenceText.textContent = `Confidence: ${Math.round(confidence * 100)}%`;
  textOutput.textContent = stableSign;

  // TTS: only speak if sign changed or cooldown elapsed
  const now = Date.now();
  if (stableSign !== lastSpokenSign || now - lastSpokenTime > COOLDOWN_MS) {
    speak(stableSign);
    lastSpokenSign = stableSign;
    lastSpokenTime = now;
    predictionBuffer = []; // reset after speaking
  }
}

// ── Text-to-Speech ────────────────────────────────────────────────────
function speak(text) {
  if (!window.speechSynthesis) {
    speakStatus.textContent = "TTS not supported in this browser";
    return;
  }
  window.speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  utt.lang = "en-IN";
  utt.rate = 0.95;
  utt.pitch = 1.0;
  utt.onstart = () => { speakStatus.textContent = `Speaking: "${text}"`; };
  utt.onend   = () => { speakStatus.textContent = ""; };
  window.speechSynthesis.speak(utt);
}

// ── Speech Recognition ────────────────────────────────────────────────
function initSpeechRecognition() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    micStatus.textContent = "Speech recognition not supported (use Chrome/Edge)";
    micBtn.disabled = true;
    return null;
  }

  const rec = new SpeechRec();
  rec.lang = "en-IN";
  rec.interimResults = true;
  rec.continuous = false;

  rec.onresult = (event) => {
    let interim = "", final = "";
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const t = event.results[i][0].transcript;
      if (event.results[i].isFinal) final += t;
      else interim += t;
    }
    transcriptEl.textContent = final || interim;
    if (final) {
      processKeyword(final.trim());
    }
  };

  rec.onerror = (e) => {
    micStatus.textContent = `Error: ${e.error}`;
    micBtn.classList.remove("active");
    recognizing = false;
  };

  rec.onend = () => {
    micBtn.classList.remove("active");
    micStatus.textContent = "Click microphone to speak";
    recognizing = false;
  };

  return rec;
}

function toggleMic() {
  if (!recognition) {
    recognition = initSpeechRecognition();
    if (!recognition) return;
  }

  if (recognizing) {
    recognition.stop();
    recognizing = false;
    micBtn.classList.remove("active");
    micStatus.textContent = "Click microphone to speak";
  } else {
    transcriptEl.textContent = "";
    recognition.start();
    recognizing = true;
    micBtn.classList.add("active");
    micStatus.textContent = "Listening... speak now";
  }
}

// ── Keyword processing ────────────────────────────────────────────────
function processKeyword(text) {
  const lower = text.toLowerCase().trim();
  let matched = null;

  for (const [key, val] of Object.entries(VOCABULARY)) {
    if (lower.includes(key)) {
      matched = val;
      break;
    }
  }

  if (matched) {
    recognizedKw.textContent = matched.label;
    recognizedKw.style.color = "#58a6ff";
    showSignVisual(matched);
  } else {
    recognizedKw.textContent = "Not found";
    recognizedKw.style.color = "#f85149";
    showUnsupported(text);
  }
}

// ── Sign visual display ────────────────────────────────────────────────
async function showSignVisual(vocab) {
  signVisualArea.innerHTML = "";

  // Try image formats
  const extensions = ["gif", "mp4", "png", "jpg", "jpeg", "webp"];
  let found = false;

  for (const ext of extensions) {
    const url = `/assets/sign_visuals/${vocab.file}.${ext}`;
    const ok = await checkAsset(url);
    if (ok) {
      found = true;
      if (ext === "mp4") {
        signVisualArea.innerHTML = `
          <div class="sign-visual-card">
            <div class="sign-visual-title">${vocab.label}</div>
            <video src="${url}" autoplay loop muted playsinline></video>
          </div>`;
      } else {
        signVisualArea.innerHTML = `
          <div class="sign-visual-card">
            <div class="sign-visual-title">${vocab.label}</div>
            <img src="${url}" alt="${vocab.label} sign" />
          </div>`;
      }
      break;
    }
  }

  if (!found) {
    // Show placeholder card with sign name
    signVisualArea.innerHTML = `
      <div class="sign-visual-card">
        <div class="sign-visual-unavailable">
          <div class="sign-name">${vocab.label.toUpperCase()}</div>
          <div style="font-size:3rem;margin:8px 0">&#9995;</div>
          <div class="sign-note">Sign visual unavailable &mdash; add approved ISL visual to assets/sign_visuals/${vocab.file}.gif</div>
        </div>
      </div>`;
  }
}

function showUnsupported(text) {
  signVisualArea.innerHTML = `
    <div class="sign-visual-card">
      <div class="sign-visual-unavailable">
        <div class="sign-name" style="font-size:1.2rem;color:#f85149">Not in vocabulary</div>
        <div style="font-size:0.85rem;color:#8b949e;margin-top:8px">
          "${text}" is not in the prototype vocabulary.<br/>
          Supported: Hello, Yes, No, Help, Thank You, Please, Sorry, Welcome, Stop, Good, Bad, Wait
        </div>
      </div>
    </div>`;
}

async function checkAsset(url) {
  try {
    const res = await fetch(url, { method: "HEAD" });
    return res.ok;
  } catch {
    return false;
  }
}

// ── Utilities ─────────────────────────────────────────────────────────
function setBadge(el, text, type) {
  el.textContent = text;
  el.className = `badge badge-${type}`;
}

function setStatus(el, text, type) {
  el.textContent = text;
  el.className = `badge badge-${type}`;
}

function clearDetection() {
  detectedSign.textContent = "—";
  confidenceBar.style.width = "0%";
  confidenceText.textContent = "Confidence: —";
  textOutput.textContent = "Waiting for sign...";
  speakStatus.textContent = "";
  predictionBuffer = [];
  lastSpokenSign = null;
}

// ── Event listeners ────────────────────────────────────────────────────
function setupEventListeners() {
  startCamBtn.addEventListener("click", startCamera);
  stopCamBtn.addEventListener("click", stopCamera);
  clearBtn.addEventListener("click", clearDetection);

  micBtn.addEventListener("click", toggleMic);

  translateBtn.addEventListener("click", () => {
    const t = textInput.value.trim();
    if (t) {
      transcriptEl.textContent = t;
      processKeyword(t);
      textInput.value = "";
    }
  });

  textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") translateBtn.click();
  });

  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      const word = chip.dataset.word;
      transcriptEl.textContent = word;
      processKeyword(word);
    });
  });
}

// ── Start ─────────────────────────────────────────────────────────────
init().catch(console.error);