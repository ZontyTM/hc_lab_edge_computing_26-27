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

    # Preprocess the image
    x = PREPROCESS(pil_img).unsqueeze(0).to(device)

    # Run the model
    logits = model(x).squeeze(0)

    # Convert logits → probabilities
    probs = torch.softmax(logits, dim=0)

    # Return prediction + confidence
    # print(probs)

    pred = int(torch.argmax(probs))
    confidence = float(torch.max(probs))
    print('pred', pred, probs[pred], confidence)

    return pred, confidence

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
    cnt = 0
    local_image_cnt = 0
    offload_image_count = 0
    sent_bytes_total = 0
    received_bytes_total = 0
    local_latencies_ms = []
    offload_latencies_ms = []
    print(args.num_samples)

    experiment_start = time.perf_counter()

    for seq in range(1, args.num_samples + 1):
        img_path = images[(seq - 1) % len(images)]
        pil_img = Image.open(img_path).convert("RGB")

        # 1) Local inference timing
        inference_start = time.perf_counter()
        pred, confidence = local_infer(model, device, pil_img)
        total_local_time = (time.perf_counter() - inference_start) * 1000
        print('total time', total_local_time)

        cnt += 1
        
        # 2) HI decision
        # TODO:
        decision = "LOCAL" if confidence >= args.tau else "OFFLOAD"
        print(decision)
        
        if decision == "LOCAL":
            # TODO:
            # - count as local
            local_image_cnt += 1

            print(pred, confidence)

            # - record local_ms
            print('total local time', total_local_time)
            local_latencies_ms.append(total_local_time)
            # - do nothing else (no network)

        elif decision == "OFFLOAD":
            # TODO:

            offload_start = time.perf_counter()
            # - count as offload
            offload_image_count += 1

            # - build OFFLOAD packet (e.g., JPEG + metadata)
            img = jpeg_for_udp(pil_img, args.jpeg_quality)
            packed_img = pack_imgd({}, img)

            # - send packet via UDP
            sock.connect(server_addr)
            sent_bytes_total += sock.send(packed_img)

            data, address = sock.recvfrom(1024)
            received_bytes_total += len(data)
            pred, confidence = tuple(json.loads(data.decode()))
            print(pred, confidence)

            offload_latency = (time.perf_counter() - offload_start) * 1000
            offload_latencies_ms.append(offload_latency)
            print('offload time', offload_latency)

        else:
            raise RuntimeError("Invalid decision")

        time.sleep(args.period)

    runtime_s = time.perf_counter() - experiment_start

    # -------------------------
    # Final report 
    # -------------------------
    offload_rate = 100 * offload_image_count / cnt if cnt else 0.0
    total_bytes = sent_bytes_total + received_bytes_total
    traffic_bytes_per_sec = total_bytes / runtime_s if runtime_s else 0.0
    local_latency = 0 if local_image_cnt == 0 else sum(local_latencies_ms) / len(local_latencies_ms)
    offload_latency = 0 if offload_image_count == 0 else sum(offload_latencies_ms) / len(offload_latencies_ms)

    print(f"\nEvaluation:")
    print(f"offload rate: {offload_rate:.2f}%")
    print(f"network traffic: {traffic_bytes_per_sec:.2f} bytes/sec")
    print(f"latency:")
    print(f"    local: {local_latency:.2f} ms")
    print(f"    offload: {offload_latency:.2f} ms")



if __name__ == "__main__":
    main()
