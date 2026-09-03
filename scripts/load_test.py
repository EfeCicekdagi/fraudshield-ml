import time
import requests
import concurrent.futures

API_URL = "http://localhost:8000"
API_KEY = "test_key"
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

payload = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 5000,
    "oldbalanceOrg": 5000,
    "oldbalanceDest": 0,
    "orig_account_type": "C",
    "dest_account_type": "C"
}

def send_request():
    start = time.time()
    try:
        response = requests.post(f"{API_URL}/api/v1/predict", json=payload, headers=HEADERS, params={"explain": "false"})
        return response.status_code, time.time() - start
    except Exception as e:
        return 500, time.time() - start

def load_test(num_requests=100, concurrency=10):
    import psutil
    import os
    print(f"Starting load test with {num_requests} requests, concurrency {concurrency}")
    
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = psutil.virtual_memory().used
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(lambda _: send_request(), range(num_requests)))
        
    total_time = time.time() - start_time
    
    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = psutil.virtual_memory().used
    
    successes = [r for r in results if r[0] == 200]
    errors = [r for r in results if r[0] != 200]
    latencies = [r[1] for r in results if r[0] == 200]
    
    status_dist = {}
    for r in results:
        status_dist[r[0]] = status_dist.get(r[0], 0) + 1
        
    latencies.sort()
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    p50_latency = latencies[int(len(latencies)*0.5)] if latencies else 0
    p95_latency = latencies[int(len(latencies)*0.95)] if latencies else 0
    p99_latency = latencies[int(len(latencies)*0.99)] if latencies else 0
    
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {num_requests / total_time:.2f} req/s")
    print(f"Successes: {len(successes)}")
    print(f"Errors: {len(errors)}")
    print(f"HTTP Status Distribution: {status_dist}")
    print(f"Avg Latency: {avg_latency*1000:.2f}ms")
    print(f"p50 Latency: {p50_latency*1000:.2f}ms")
    print(f"p95 Latency: {p95_latency*1000:.2f}ms")
    print(f"p99 Latency: {p99_latency*1000:.2f}ms")
    print(f"Max Latency: {max_latency*1000:.2f}ms")
    print(f"Host CPU (approx): {cpu_after}%")
    print(f"Host Memory Delta (approx): {(mem_after - mem_before) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    c = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    load_test(n, c)
