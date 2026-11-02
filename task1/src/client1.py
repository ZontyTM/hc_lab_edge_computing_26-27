"""
Task 1 — Telemetry Client (SKELETON)
===================================

You implement a UDP-based telemetry client with Hierarchical Inference (HI).

For each telemetry sample:
1) run local inference using json_small.joblib
2) compute confidence (max predicted probability)
3) decide:
   - LOCAL  if confidence >= tau
   - OFFLOAD if confidence < tau
4) ONLY IF OFFLOAD:
   - send the sample to the server via UDP
5) Measure and report:
   - offload rate (% of samples sent to server)
   - network traffic (bytes/sec and total bytes)
   - latency:
       * local inference latency
       * offloaded end-to-end latency
You are free to:
- print results
- store them in files
- plot them
"""

import argparse
import socket




def main():
    # -------------------------
    # Parse arguments
    # -------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-host", default="localhost")
    ap.add_argument("--server-port", type=int, default=10001)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--num-samples", type=int, default=100)
    args = ap.parse_args()

    # -------------------------
    # Load data and model
    # -------------------------
    # TODO:
    # - load telemetry.jsonl
    # - load schema.json
    # - load json_small.joblib

    # -------------------------
    # Setup UDP socket
    # -------------------------
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_addr = (args.server_host, args.server_port)

    # -------------------------
    # Measurement variables
    # -------------------------
    # TODO:
    # - counters for total samples / offloaded samples
    # - byte counters for sent / received data
    # - lists for local latency and offload latency

    for i in range(args.num_samples):
        # -------------------------
        # 1) Prepare sample
        # -------------------------
        # TODO:
        # - read one telemetry JSON sample
        # - convert to feature vector (schema order!)

        # -------------------------
        # 2) Local inference
        # -------------------------
        # TODO: run small model predict_proba

        # -------------------------
        # 3) Compute confidence
        # -------------------------
        # TODO:
        # confidence = max predicted probability

        # -------------------------
        # 4) HI decision
        # -------------------------
        # TODO

    # -------------------------
    # Final report
    # -------------------------
    # TODO:
    # - compute offload rate
    # - compute latencies
    # - compute traffic statistics
    # - save results
        pass
 

if __name__ == "__main__":
    main()
