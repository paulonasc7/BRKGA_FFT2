import sys
import time

# Thread test with 4 workers
sys.argv = ['BRKGA_alg3.py', '50', '2', '0', 'torch_gpu', 'thread', '4']

with open('BRKGA_alg3.py') as f:
    code = f.read()
code = code.replace('num_generations = 30', 'num_generations = 3')

print("Testing thread parallelization (4 workers)...")
start = time.time()
exec(code)
print(f"\nTotal execution time: {time.time()-start:.2f}s")
