import time
import psutil
import pandas as pd
import numpy as np
import random
import os
import json
from fraudshield.inference.predictor import FraudPredictor
from fraudshield.inference.schemas import TransactionRequest, BatchTransactionRequest
import tempfile
import yaml

def generate_synthetic_data(n=1000):
    np.random.seed(42)
    random.seed(42)
    types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
    data = []
    for _ in range(n):
        t_type = random.choice(types)
        amount = max(0.0, np.random.normal(150000, 200000))
        oldbalanceOrg = max(0.0, np.random.normal(50000, 100000))
        if random.random() < 0.1: amount = 0.0
        if random.random() < 0.1: oldbalanceOrg = 0.0
        data.append({
            "step": random.randint(1, 744),
            "type": t_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "nameOrig": f"C{random.randint(1000, 9000)}",
            "nameDest": f"M{random.randint(1000, 9000)}" if t_type == 'PAYMENT' else f"C{random.randint(1000, 9000)}"
        })
    return data

def run_benchmark():
    results = {}
    
    # Base config
    with open("configs/inference.yaml", "r") as f:
        base_config = yaml.safe_load(f)
        
    def create_predictor(explain=False):
        config = base_config.copy()
        if "explainability" not in config:
            config["explainability"] = {}
        config["explainability"]["enabled"] = explain
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".yaml") as tmp:
            yaml.dump(config, tmp)
            tmp_path = tmp.name
            
        return FraudPredictor(config_path=tmp_path)
    
    print("Measuring cold load time...")
    start_time = time.perf_counter()
    predictor_no_exp = create_predictor(explain=False)
    end_time = time.perf_counter()
    results["cold_load_time_ms"] = float(f"{(end_time - start_time) * 1000:.2f}")
    
    predictor_exp = create_predictor(explain=True)
    
    transactions = generate_synthetic_data(1000)
    requests = [TransactionRequest(**tx) for tx in transactions]
    
    def measure_single(predictor, reqs):
        predictor.predict_single(reqs[0]) # warmup
        latencies = []
        for req in reqs:
            start_time = time.perf_counter()
            _ = predictor.predict_single(req)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
        return float(f"{np.percentile(latencies, 50):.2f}"), float(f"{np.percentile(latencies, 95):.2f}")
        
    print("Measuring warm single latency (no explanation)...")
    p50_no, p95_no = measure_single(predictor_no_exp, requests[:100])
    results["no_explanation_p50_ms"] = p50_no
    results["no_explanation_p95_ms"] = p95_no
    
    print("Measuring warm single latency (with Integrated Gradients)...")
    p50_exp, p95_exp = measure_single(predictor_exp, requests[:100])
    results["integrated_gradients_p50_ms"] = p50_exp
    results["integrated_gradients_p95_ms"] = p95_exp
    
    def measure_batch(predictor, bs):
        batch_reqs = requests[:bs]
        batch_obj = BatchTransactionRequest(transactions=batch_reqs)
        predictor.predict_batch(batch_obj) # warmup
        start_time = time.perf_counter()
        _ = predictor.predict_batch(batch_obj)
        end_time = time.perf_counter()
        return float(f"{bs / (end_time - start_time):.2f}")
        
    print("Measuring batch throughput (no explanation)...")
    results["batch_throughput_without_explanations"] = measure_batch(predictor_no_exp, 1000)
    
    print("Measuring batch throughput (with explanations)...")
    results["batch_throughput_with_explanations"] = measure_batch(predictor_exp, 100) # smaller batch due to time
    
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    results["memory_footprint_mb"] = float(f"{mem_mb:.2f}")
    print(f"Memory footprint: {mem_mb:.2f} MB")
    
    out_file = "artifacts/final/inference_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Benchmark completed. Results saved to {out_file}")

if __name__ == "__main__":
    run_benchmark()
