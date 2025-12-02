from flask import Flask, request, jsonify
import torch
import numpy as np
import os

# PointerNet 모델을 가져옴 (TransformerPointer 아님)
from model_attention import PointerNet

app = Flask(__name__)

# -----------------------------
# Load Model
# -----------------------------
device = "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "attn_pointer_rl.pt")

print(">>> Loading model from:", MODEL_PATH)
print(">>> Files in directory:", os.listdir(BASE_DIR))

# Colab에서 학습한 정확한 모델(=PointerNet)
model = PointerNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# -----------------------------
# Utilities
# -----------------------------
def infer_best_order(coords):
    """coords: numpy array (N,2)"""

    coords_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        actions, _ = model(coords_tensor)   # PointerNet returns (actions, log_probs)

    order = actions.squeeze(0).tolist()
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
    data = request.get_json()
    coords = np.array(data["coords"])

    order = infer_best_order(coords)

    return jsonify({
        "order": order
    })


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
