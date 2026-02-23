import subprocess
import re
import json
import matplotlib.pyplot as plt

def run_benchmark(policy, seq_len, threshold=0.1):
    print(f"Running benchmark with policy={policy}, seq_len={seq_len}...")
    cmd = [
        "python", "benchmark_large_batch.py",
        "--policy", policy,
        "--seq_len", str(seq_len),
        "--main_size", "128"
    ]
    if policy == "markov":
        cmd.extend(["--admission_threshold", str(threshold)])
    
    # Needs HF_ENDPOINT for huggingface download
    env = {"HF_ENDPOINT": "https://hf-mirror.com"}
    import os
    merged_env = os.environ.copy()
    merged_env.update(env)
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=merged_env)
    
    # Parse the output
    # Example output to parse:
    # 总命中 (Hits): 2048
    # 总未命中 (Misses): 2047
    
    hits = 0
    misses = 0
    bypassed = 0
    
    for line in result.stdout.split('\n'):
        if "总命中 (Hits):" in line:
            hits = int(re.search(r'\d+', line).group())
        elif "总未命中 (Misses):" in line:
            misses = int(re.search(r'\d+', line).group())
        elif "被概率滤镜拦截 (Bypassed):" in line:
            bypassed = int(re.search(r'\d+', line).group())
            
    if hits == 0 and misses == 0:
        print(f"Error parsing output or crash. Output:\n{result.stdout}\n{result.stderr}")
        return None
        
    return {
        "hits": hits,
        "misses": misses,
        "bypassed": bypassed,
        "miss_rate": misses / (hits + misses) if hits + misses > 0 else 0
    }

def main():
    batch_sizes = [128, 256, 512, 1024, 2048, 4096]
    lru_misses = []
    markov_misses = []
    
    results = {}
    
    for b in batch_sizes:
        # Run LRU
        lru_res = run_benchmark("lru", b)
        if lru_res:
            lru_misses.append(lru_res["misses"])
        else:
            lru_misses.append(0)
            
        # Run Markov (Admission Control)
        markov_res = run_benchmark("markov", b, threshold=0.1)
        if markov_res:
            markov_misses.append(markov_res["misses"])
        else:
            markov_misses.append(0)
            
        results[b] = {
            "lru": lru_res,
            "markov": markov_res
        }
        
    print("\n=== 最终测试结果 ===")
    print(json.dumps(results, indent=2))
    
    # Save results to file
    with open("regime_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # We will plot using a separate script later if needed, 
    # but let's print a CSV format for easy copy-pasting
    print("\nBatchSize,LRU_Misses,Markov_Misses")
    for i, b in enumerate(batch_sizes):
        print(f"{b},{lru_misses[i]},{markov_misses[i]}")

if __name__ == "__main__":
    main()
