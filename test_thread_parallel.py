import sys
import time

# Test serial vs thread modes
for eval_mode in ["serial", "thread"]:
    for workers in [1, 2, 4]:
        if eval_mode == "serial" and workers > 1:
            continue
            
        sys.argv = ['BRKGA_alg3.py', '50', '2', '0', 'torch_gpu', eval_mode, str(workers)]
        
        with open('BRKGA_alg3.py') as f:
            code = f.read()
        code = code.replace('num_generations = 30', 'num_generations = 3')
        
        print(f"\n{'='*50}")
        print(f"Testing: eval_mode={eval_mode}, workers={workers}")
        print('='*50)
        
        start = time.perf_counter()
        exec(code)
        elapsed = time.perf_counter() - start
        print(f"Total time: {elapsed:.2f}s, Per generation: {elapsed/3:.2f}s")
