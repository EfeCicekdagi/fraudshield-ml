import time
import requests
import json
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import argparse

API_URL = "http://127.0.0.1:8000/api/v1"
HEADERS = {"X-API-Key": "test-secret-key"}

def generate_payload(batch_size=1):
    txs = []
    for _ in range(batch_size):
        txs.append({
            "step": np.random.randint(1, 744),
            "type": np.random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]),
            "amount": float(np.random.uniform(10, 10000)),
            "oldbalanceOrg": float(np.random.uniform(0, 50000)),
            "orig_account_type": "C",
            "dest_account_type": "C"
        })
    if batch_size == 1:
        return txs[0]
    return {"transactions": txs}

def measure_latency(endpoint, payload, explain=False, iterations=100):
    latencies = []
    url = f"{API_URL}/{endpoint}"
    if explain:
        url += "?explain=true"
        
    for _ in range(iterations):
        t0 = time.time()
        res = requests.post(url, json=payload, headers=HEADERS)
        t1 = time.time()
        if res.status_code == 200:
            latencies.append((t1 - t0) * 1000)
            
    if not latencies:
        return 0, 0
    return np.median(latencies), np.percentile(latencies, 95)

def worker_predict(args):
    payload = generate_payload(1)
    t0 = time.time()
    res = requests.post(f"{API_URL}/predict", json=payload, headers=HEADERS)
    t1 = time.time()
    return (t1 - t0) * 1000 if res.status_code == 200 else None

def benchmark_concurrent(workers, iterations=100):
    with Pool(workers) as p:
        args = [None] * iterations
        t0 = time.time()
        latencies = p.map(worker_predict, args)
        t1 = time.time()
        
    valid_latencies = [l for l in latencies if l is not None]
    total_time = t1 - t0
    throughput = len(valid_latencies) / total_time
    
    return np.median(valid_latencies), throughput

def main():
    print("=== FraudShield API Benchmark ===")
    print("Methodology: Localhost networking via Python `requests` library.")
    print("Note: Concurrency limits (e.g. semaphore) are per-process.")
    
    # 1. Health & Startup
    try:
        t0 = time.time()
        res = requests.get(f"http://127.0.0.1:8000/health/live")
        t1 = time.time()
        if res.status_code != 200:
            print("API /health/live failed. Start it first with 'fraudshield serve'")
            return
        print(f"\n/health/live Latency (Process check): {(t1-t0)*1000:.2f} ms")
        
        t0 = time.time()
        res = requests.get(f"http://127.0.0.1:8000/health/ready")
        t1 = time.time()
        if res.status_code != 200:
            print("API /health/ready failed. Predictor not initialized.")
            return
        print(f"/health/ready Latency (In-memory check): {(t1-t0)*1000:.2f} ms")
    except requests.exceptions.ConnectionError:
        print("API is not running. Start it first with 'fraudshield serve'")
        return

    # 2. Single Predict Latency
    payload_single = generate_payload(1)
    p50, p95 = measure_latency("predict", payload_single, explain=False, iterations=100)
    print(f"\n/predict (No Explain): p50={p50:.2f} ms, p95={p95:.2f} ms")
    
    # 3. Single Explain Latency
    p50_exp, p95_exp = measure_latency("explain", payload_single, explain=False, iterations=50)
    print(f"/explain (IG Enabled): p50={p50_exp:.2f} ms, p95={p95_exp:.2f} ms")
    
    # 4. Batch Throughput
    batch_size = 100
    payload_batch = generate_payload(batch_size)
    
    t0 = time.time()
    res = requests.post(f"{API_URL}/predict/batch", json=payload_batch, headers=HEADERS)
    t1 = time.time()
    
    if res.status_code == 200:
        batch_lat = (t1-t0)*1000
        total_time = t1 - t0
        req_sec = 1 / total_time
        tx_sec = batch_size / total_time
        print(f"\n/predict/batch (Size 100): {batch_lat:.2f} ms")
        print(f"Batch Throughput: {req_sec:.2f} HTTP requests/sec | {tx_sec:.2f} transactions/sec")
    
    # 5. Concurrent Requests
    print("\nConcurrent Performance (Single Predict without Explain):")
    for w in [1, 4, 8, 16]:
        med, thr = benchmark_concurrent(w, iterations=200)
        print(f"Workers: {w} | Median Latency: {med:.2f} ms | Throughput: {thr:.2f} req/sec")
        
    print("\nBenchmark Finished.")

if __name__ == "__main__":
    main()
