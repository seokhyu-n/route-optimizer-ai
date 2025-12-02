from flask import Flask, request, jsonify
import torch
import numpy as np
import os
from model_attention import TransformerPointer

app = Flask(__name__)

# -----------------------------
# Load Model with absolute path
# -----------------------------
device = "cpu"

# Current directory of this file (Render에서 /opt/render/project/src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path
MODEL_PATH = os.path.join(BASE_DIR, "attn_pointer_rl.pt")

print(">>> [DEBUG] BASE_DIR:", BASE_DIR)
print(">>> [DEBUG] MODEL_PATH:", MODEL_PATH)
print(">>> [DEBUG] Directory contents:", os.listdir(BASE_DIR))

# Load model
model = TransformerPointer()
try:
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    print(">>> [DEBUG] Model loaded successfully!")
except Exception as e:
    print(">>> [ERROR] Failed to load model:", e)

model.eval()


# -----------------------------
# Utilities
# -----------------------------
def infer_best_order(coords):
    """coords: numpy array (N,2) shape"""

    coords_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        scores = model(coords_tensor)  # (1, N)
        scores = scores.squeeze(0)

    order = torch.argsort(scores, descending=False).tolist()
    return order


# -----------------------------
# Root
# -----------------------------
@app.route("/")
def home():
    return "Hello! Route Optimizer AI is running on Render."


# -----------------------------
# API: /optimize
# -----------------------------
@app.route("/optimize", methods=["POST"])
def optimize():
    try:
        data = request.get_json()

        if "coords" not in data:
            return jsonify({"error": "coords field missing"}), 400

        coords = np.array(data["coords"], dtype=float)

        if coords.ndim != 2 or coords.shape[1] != 2:
            return jsonify({"error": "coords must be N x 2 array"}), 400

        order = infer_best_order(coords)

        return jsonify({"order": order})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Run (local only)
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
