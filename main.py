from flask import Flask, request, jsonify
import numpy as np
import torch
from model_attention import load_model

app = Flask(__name__)

model = load_model("attn_pointer_rl.pt", device="cpu")

def get_route(coords):
    coords = np.array(coords, dtype=np.float32)
    coords = torch.tensor(coords).unsqueeze(0)
    logits = model(coords)
    order = torch.argsort(logits, dim=-1).squeeze(0).tolist()
    return order

@app.route("/optimize", methods=["POST"])
def optimize():
    data = request.json
    coords = data["coords"]
    order = get_route(coords)
    return jsonify({"order": order})

@app.route("/")
def home():
    return "Route Optimizer API Ready!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
