"""
Task 1 — Image Client (SKELETON)
=====================================================================

You implement a UDP client that:
1) iterates over images in data/images/
2) runs local inference with MobileNetV3-Small 
3) computes confidence = MSP = max softmax probability
4) HI decision:
   - LOCAL   if conf >= tau  -> do NOT send anything
   - OFFLOAD if conf <  tau  -> send image (+ metadata) to server via UDP

Measure and report :
- offload rate (% offloaded)
- network traffic (bytes/sec and total bytes)
- latency:
    * local inference latency (ms)
    * offloaded end-to-end latency (ms)  [only possible if your server replies]

Notes:
- For OFFLOAD, transmit a compressed JPEG to reduce UDP payload size.
- UDP may reorder packets; therefore a SINGLE combined packet for OFFLOAD is recommended.

(Optional, recommended):
- server replies with a RESULT message so you can measure offload RTT.
"""

import argparse
import json
import socket
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
from PIL import Image
from torchvision import models, transforms


# -------------------------
# GIVEN: model + preprocessing
# -------------------------

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_local_model(device: str):
    """
    GIVEN: Local model.
    Torchvision downloads pretrained weights automatically on first run.
    """
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    ).to(device).eval()
    return model


@torch.no_grad()
def local_infer(model, device: str, pil_img: Image.Image) -> Tuple[int, float]:
    """
    TODO:
    This function should:
    -Preprocess the image
    -Run the model
    -Convert logits → probabilities
    -Return prediction + confidence
   
    """
    raise NotImplementedError


def jpeg_for_udp(pil_img: Image.Image, quality: int = 75) -> bytes:
    """
    GIVEN: JPEG compression to reduce UDP payload size.
    """
    img = pil_img.convert("RGB").resize((224, 224))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# -------------------------
# GIVEN: packet helpers 
# -------------------------
# You may define your own protocol, but make sure the server can parse it.

def pack_json(magic: bytes, obj: Dict) -> bytes:
    payload = json.dumps(obj).encode("utf-8")
    return magic + len(payload).to_bytes(4, "big", signed=False) + payload


def pack_imgd(meta: Dict, jpeg_bytes: bytes) -> bytes:
    """
    OFFLOAD packet (single UDP datagram):
    b'IMGD' + meta_len(4) + meta_json + jpeg_len(4) + jpeg_bytes
    """
    meta_bytes = json.dumps(meta).encode("utf-8")
    return (
        b"IMGD"
        + len(meta_bytes).to_bytes(4, "big", signed=False)
        + meta_bytes
        + len(jpeg_bytes).to_bytes(4, "big", signed=False)
        + jpeg_bytes
    )

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-host", default="localhost")
    ap.add_argument("--server-port", type=int, default=10001)

    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--period", type=float, default=0.05)

    ap.add_argument("--image-dir", type=str, default="data/images")
    ap.add_argument("--jpeg-quality", type=int, default=75)

    ap.add_argument("--wait-reply", action="store_true",
                    help="If set: wait for server reply for OFFLOAD to measure RTT.")
    ap.add_argument("--timeout", type=float, default=1.0)
    args = ap.parse_args()

    server_addr = (args.server_host, args.server_port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Load images
    img_dir = Path(args.image_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise ValueError(f"No images found in: {img_dir}")

    # Load model 
    device = get_device()
    model = load_local_model(device)

    # --------------
    # Measurements 
    # --------------
    # TODO:
    # - total_samples counter
    # - offloaded_samples counter
    # - sent_bytes_total / received_bytes_total
    # - store local inference latencies 

    start_time = time.perf_counter()

    for seq in range(1, args.num_samples + 1):
        img_path = images[(seq - 1) % len(images)]
        pil_img = Image.open(img_path).convert("RGB")

        # 1) Local inference timing
        # TODO: # Measure how long local inference takes 

        # 2) HI decision
        # TODO:
        decision = None  # placeholder
        
        if decision == "LOCAL":
            # TODO:
            # - count as local
            # - record local_ms
            # - do nothing else (no network)
            pass

        elif decision == "OFFLOAD":
            # TODO:
            # - count as offload
            # - build OFFLOAD packet (e.g., JPEG + metadata)
            # - send packet via UDP
            # - measure offload latency if you wait for server reply
            pass

        else:
            raise RuntimeError("Invalid decision")

        time.sleep(args.period)

    runtime_s = time.perf_counter() - start_time

    # -------------------------
    # Final report 
    # -------------------------
    # TODO:
    # - offload_rate 
    # - traffic
    # - latency



if __name__ == "__main__":
    main()
