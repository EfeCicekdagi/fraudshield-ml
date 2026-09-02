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
    "nameOrig": "C12345",
    "oldbalanceOrg": 5000,
    "newbalanceOrig": 0,
    "nameDest": "C67890",
    "oldbalanceDest": 0,
    "newbalanceDest": 5000
}

def send_request():
    start = time.time()
    try:
        response = requests.post(f"{API_URL}/api/v1/predict", json=payload, headers=HEADERS, params={"explain": "false"})
        return response.status_code, time.time() - start
    except Exception as e:
        return 500, time.time() - start

def load_test(num_requests=100, concurrency=10):
    print(f"Starting load test with {num_requests} requests, concurrency {concurrency}")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(lambda _: send_request(), range(num_requests)))
        
    total_time = time.time() - start_time
    
    successes = [r for r in results if r[0] == 200]
    errors = [r for r in results if r[0] != 200]
    latencies = [r[1] for r in results if r[0] == 200]
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {num_requests / total_time:.2f} req/s")
    print(f"Successes: {len(successes)}")
    print(f"Errors: {len(errors)}")
    print(f"Avg Latency: {avg_latency*1000:.2f}ms")
    print(f"Max Latency: {max_latency*1000:.2f}ms")

if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    c = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    load_test(n, c)
