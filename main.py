import os
import json
import numpy as np
import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Paths ──────────────────────────────────────────────────────────
MODEL_PATH = "model/sign_model.pkl"
ENCODER_PATH = "model/label_encoder.pkl"
STATIC_DIR = "static"
ASSETS_DIR = "assets"

# ── App ────────────────────────────────────────────────────────────
app = FastAPI(title="SignBridge AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount static dirs ──────────────────────────────────────────────
# Ensure these exist so a missing folder (e.g. no sign visuals added yet)
# never crashes server startup.
os.makedirs(os.path.join(ASSETS_DIR, "sign_visuals"), exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

# ── Load model ─────────────────────────────────────────────────────
clf = None
label_encoder = None

def load_model():
    global clf, label_encoder
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        try:
            clf = joblib.load(MODEL_PATH)
            label_encoder = joblib.load(ENCODER_PATH)
            print(f"[SignBridge AI] Model loaded: {list(label_encoder.classes_)}")
            return True
        except Exception as e:
            print(f"[SignBridge AI] Model load error: {e}")
            clf = None
            label_encoder = None
    return False

load_model()

# ── Normalization (must match collect_data.py) ─────────────────────
def normalize_landmarks(raw_landmarks: list) -> np.ndarray:
    """
    raw_landmarks: flat list of 63 floats [x0,y0,z0, x1,y1,z1, ...]
    Returns normalized flat array of 63 floats.
    """
    lm = np.array(raw_landmarks, dtype=np.float32).reshape(21, 3)
    wrist = lm[0].copy()
    lm -= wrist
    distances = np.linalg.norm(lm, axis=1)
    max_dist = distances.max()
    if max_dist > 0:
        lm /= max_dist
    return lm.flatten()


# ── Routes ─────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/health")
async def health():
    model_loaded = clf is not None and label_encoder is not None
    classes = list(label_encoder.classes_) if label_encoder else []
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "classes": classes,
        "model_path": MODEL_PATH,
        "message": "SignBridge AI is running"
    }

@app.post("/predict")
async def predict_http(payload: dict):
    """HTTP fallback prediction endpoint."""
    if clf is None or label_encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run python train_model.py first."
        )
    landmarks = payload.get("landmarks")
    if not landmarks or len(landmarks) != 63:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 63 landmark values, got {len(landmarks) if landmarks else 0}"
        )
    try:
        features = normalize_landmarks(landmarks).reshape(1, -1)
        probs = clf.predict_proba(features)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        label = label_encoder.inverse_transform([pred_idx])[0]
        return {
            "sign": label,
            "confidence": round(confidence, 4),
            "all_probs": {
                label_encoder.classes_[i]: round(float(probs[i]), 4)
                for i in range(len(probs))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")

    if clf is None or label_encoder is None:
        await websocket.send_json({
            "type": "error",
            "message": "Model not loaded. Run python train_model.py first."
        })
        await websocket.close()
        return

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            landmarks = data.get("landmarks")
            if not landmarks:
                await websocket.send_json({"type": "error", "message": "No landmarks"})
                continue

            if len(landmarks) != 63:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Expected 63 values, got {len(landmarks)}"
                })
                continue

            try:
                features = normalize_landmarks(landmarks).reshape(1, -1)
                probs = clf.predict_proba(features)[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                label = label_encoder.inverse_transform([pred_idx])[0]

                await websocket.send_json({
                    "type": "prediction",
                    "sign": label,
                    "confidence": round(confidence, 4),
                    "all_probs": {
                        label_encoder.classes_[i]: round(float(probs[i]), 4)
                        for i in range(len(probs))
                    }
                })
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)