# SignBridge AI
### Smart India Hackathon MVP — Real-Time Two-Way Sign Language Communication

> **IMPORTANT DISCLAIMER**: This is a limited prototype system demonstrating
> a vocabulary of 12 signs. It does not translate complete Indian Sign Language.

---

## Quick Start

### 0. Check your Python version (Windows)
```bash
python --version
```
MediaPipe (used only by `collect_data.py`) currently supports **Python 3.9–3.12
only** — it has no installable wheels for 3.13/3.14. If your `python --version`
reports 3.13 or higher:
1. Install Python 3.12 from python.org (this can coexist with your current version).
2. Create a virtual environment with it: `py -3.12 -m venv venv`
3. Activate it: `venv\Scripts\activate`
4. Continue with the steps below inside that environment.

If you're already on 3.9–3.12, skip straight to step 1.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Collect Training Data
```bash
python collect_data.py
```
When the webcam opens, perform each sign gesture when prompted.
- 12 classes: Hello, Yes, No, Help, Thank You, Please, Sorry, Welcome, Stop, Good, Bad, Wait
- 60 samples per class (720 total)
- The script resumes from wherever `data/landmarks.csv` left off — it will
  skip any class that already has 60+ samples, so re-running it is safe.
- Press **Q** to quit early

A dataset with 60 samples/class for all 12 signs (720 rows) is already
included in `data/landmarks.csv` from a previous collection run, and a model
trained on it already exists in `model/`. You don't have to re-collect or
re-train to run the demo — only do so if you want to add more samples or
improve accuracy.

### 3. Train the Model
```bash
python train_model.py
```
Outputs real accuracy/classification report and saves the model to
`model/sign_model.pkl` (+ `model/label_encoder.pkl`).

### 4. Run the Application
```bash
python -m uvicorn main:app --reload --port 8000
```
(Using `python -m uvicorn ...` instead of a bare `uvicorn` command avoids
"uvicorn is not recognized" errors on Windows when the Scripts folder isn't
on your PATH. Alternatively, just run `python main.py` — it starts the same
server directly.)

Open: https://sign-bridge-ai-gold.vercel.app/

---

## Project Structure
```
SignBridge AI/
├── data/landmarks.csv          # Collected training data (label + 63 landmark features)
├── model/
│   ├── sign_model.pkl          # Trained RandomForest classifier
│   ├── label_encoder.pkl       # Label encoder
│   └── hand_landmarker.task    # MediaPipe hand landmark model (used by collect_data.py)
├── assets/sign_visuals/        # Sign GIF/image assets (add your own; UI falls back
│                                # to a placeholder card if a file is missing)
├── static/
│   ├── index.html              # Frontend
│   ├── style.css               # Dark-themed UI
│   └── app.js                  # MediaPipe + WebSocket + Speech
├── collect_data.py             # Data collection tool
├── train_model.py              # Model training
├── main.py                     # FastAPI backend
└── requirements.txt
```

## Technology Stack
| Layer | Technology |
|-------|------------|
| Backend | Python 3.9–3.12, FastAPI, uvicorn |
| ML | scikit-learn RandomForestClassifier |
| Hand Detection | MediaPipe 1.0.1 (Python, offline collection), MediaPipe Tasks Vision (browser JS, live demo) |
| Computer Vision | OpenCV (data collection only) |
| Frontend | HTML5, CSS3, Vanilla JS |
| Browser APIs | Web Speech API (STT + TTS), WebSocket, getUserMedia |

## How It Works

### Sign → Text + Speech
1. Webcam captures video in the browser
2. MediaPipe Tasks Vision (WASM) detects hand landmarks in real-time — no
   frames are uploaded, only the small landmark vector
3. Normalized landmarks (63 values) sent via WebSocket to Python backend
4. RandomForest classifier predicts the sign + confidence
5. Stabilization filter (15-frame consensus + confidence threshold) prevents
   flickering and repeated speech
6. Detected sign displayed as text + spoken once via TTS

### Speech/Text → Sign
1. Web Speech API captures microphone input (Chrome/Edge), or
2. User types a word / clicks a vocabulary chip
3. Keyword matching against the 12-word vocabulary
4. Corresponding sign visual displayed, with a clear "not in prototype
   vocabulary" message for unsupported words
5. Placeholder card shown if no visual file exists yet for that word

## Adding Sign Visuals
Place GIF, PNG, MP4, JPG, or WEBP files in `assets/sign_visuals/`, named
after the vocabulary key:
```
hello.gif    yes.gif     no.gif      help.gif
thank_you.gif  please.gif  sorry.gif   welcome.gif
stop.gif     good.gif    bad.gif     wait.gif
```
No visual files are bundled — do not source or claim official ISL footage
without verifying it yourself. Until you add real files, the UI shows a
clean placeholder card with the sign name so the flow still works end-to-end.

## Supported Signs (Prototype)
Hello · Yes · No · Help · Thank You · Please · Sorry · Welcome · Stop · Good · Bad · Wait

This is a limited prototype vocabulary, not full Indian Sign Language.

## Requirements
- Python 3.9–3.12 (MediaPipe does not support 3.13/3.14 yet)
- Webcam
- Chrome or Edge (for speech recognition + WebSocket + MediaPipe WASM)
- Internet connection on first load (downloads the MediaPipe WASM hand
  model, ~8MB, from Google's CDN)
