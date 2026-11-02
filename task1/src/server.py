"""
Task 1 — UDP Server (SKELETON) 
=======================================================

You implement a UDP server that receives OFFLOAD samples from clients:

- Telemetry OFFLOAD: client sends a DATA packet (JSON) including a feature vector.
  Server runs json_large.joblib.

- Image OFFLOAD: client sends an IMGD packet containing metadata + JPEG bytes.
  Server runs ResNet-18.

Protocol:
- DATA: b'DATA' + len4 + JSON
- IMGD: b'IMGD' + meta_len4 + meta_json + blob_len4 + jpeg_bytes

You are free to log to files or only print summaries.
"""

import argparse
import json
import socket
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

DATA_MAGIC = b"DATA"
IMGD_MAGIC = b"IMGD"
RESU_MAGIC = b"RESU"


# ----------------------
# GIVEN: packet helpers
# ----------------------

def pack_json(magic: bytes, obj: Dict) -> bytes:
    payload = json.dumps(obj).encode("utf-8")
    return magic + len(payload).to_bytes(4, "big", signed=False) + payload


def parse_len4(b: bytes, offset: int) -> int:
    return int.from_bytes(b[offset:offset + 4], "big", signed=False)


def parse_data(packet: bytes) -> Dict:
    if len(packet) < 8:
        raise ValueError("packet too short")
    declared = parse_len4(packet, 4)
    payload = packet[8:]
    if declared != len(payload):
        raise ValueError("DATA length mismatch")
    return json.loads(payload.decode("utf-8"))


def parse_imgd(packet: bytes) -> Tuple[Dict, bytes]:
    """
    IMGD packet:
      0..3   b'IMGD'
      4..7   meta_len
      8..    meta_json
      ...    blob_len
      ...    blob_bytes (JPEG)
    """
    if len(packet) < 12:
        raise ValueError("IMGD packet too short")

    meta_len = parse_len4(packet, 4)
    meta_start = 8
    meta_end = meta_start + meta_len
    if meta_end + 4 > len(packet):
        raise ValueError("IMGD meta length out of bounds")

    meta = json.loads(packet[meta_start:meta_end].decode("utf-8"))

    blob_len = parse_len4(packet, meta_end)
    blob_start = meta_end + 4
    blob_end = blob_start + blob_len
    if blob_end != len(packet):
        raise ValueError("IMGD blob length mismatch")

    blob = packet[blob_start:blob_end]
    return meta, blob


# ----------------------
# telemetry helper
# ----------------------

def vec_to_np(sample_vec: List[float]) -> np.ndarray:
    return np.array(sample_vec, dtype=np.float32).reshape(1, -1)


# ----------------------
# image helper
# ----------------------

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def infer_resnet18(model, device: str, pil_img: Image.Image) -> Tuple[int, float]:
    """
    Server-side image inference using ResNet-18.
    - preprocess image
    - run model
    - convert logits to probabilities
    - return prediction and confidence
    """
    x = PREPROCESS(pil_img).unsqueeze(0).to(device)   
    logits = model(x).squeeze(0)                     
    probs = torch.softmax(logits, dim=0)
    pred = int(torch.argmax(probs).item())
    conf = float(torch.max(probs).item())
    return pred, conf

# ----------------------
# Main
# ----------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=10001)

    ap.add_argument("--model-large", type=str, default="models/json_large.joblib")
    ap.add_argument("--save-images", action="store_true",
                    help="Optional: save offloaded JPEGs under logs/server_images/")
    ap.add_argument("--log-dir", type=str, default="logs")  # optional
    args = ap.parse_args()

    # Load models 
    large_model = joblib.load(args.model_large) 

    device = get_device()
    img_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device).eval()

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(0.5)

    # Optional image save directory
    log_dir = Path(args.log_dir)
    img_dir = log_dir / "server_images"

    print(f"[SERVER] Listening udp://{args.host}:{args.port} device={device}", file=sys.stderr)

    # -------------
    # Measurements 
    # -------------
    # TODO:
    # - total_offloads counter
    # - offload_json counter
    # - offload_img counter
    # - server load

    try:
        while True:
            try:
                packet, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue

            magic = packet[:4]

            if magic == DATA_MAGIC:
                # TODO:
                pass

            elif magic == IMGD_MAGIC:
                # TODO:
                pass

            else:
                # ignore unknown packets
                pass

    except KeyboardInterrupt:
        print("\n[SERVER] Stopped.", file=sys.stderr)
    finally:
        sock.close()



if __name__ == "__main__":
    main()
