"""
Detailed profiler that breaks down time within each phase of wave_batch_evaluator.
Measures: Python loop overhead, numpy ops, torch ops, vacancy checks, tuple comparisons, etc.
"""
import sys
import numpy as np
import torch
import time
from collections import defaultdict

# Reuse the setup from profile_phases.py
sys.path.insert(0, '.')
from profile_quick import setup_problem
from wave_batch_evaluator import WaveBatchEvaluator

def profile_decode_sequences(evaluator, chromosomes, machine_idx, n_iters=5):
    """Profile _decode_sequences independently."""
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        seqs = evaluator._decode_sequences(chromosomes, machine_idx)
        times.append(time.perf_counter() - t0)
    return np.mean(times), seqs

def profile_generation_breakdown(evaluator, chromosomes, n_gens=3):
    """Profile one full evaluate_batch call with detailed breakdown."""
    # Warm up
    evaluator.evaluate_batch(chromosomes)

    # Now profile
    gen_times = []
    for g in range(n_gens):
        t0 = time.perf_counter()
        evaluator.evaluate_batch(chromosomes)
        gen_times.append(time.perf_counter() - t0)

    return gen_times

def profile_phase3_breakdown(evaluator, chromosomes, machine_idx):
    """
    Manually step through one wave to measure Phase 3 sub-costs:
    - Python loop overhead (iteration, attribute lookups)
    - Vacancy check time (numba calls)
    - List append time
    """
    from numba_utils import check_vacancy_fit_simple

    mach_data = evaluator.machines[machine_idx]
    H, W = mach_data.bin_length, mach_data.bin_width

    # Set up contexts (run a few waves first to get bins open)
    num_solutions = len(chromosomes)
    max_bins_per_sol = max(10, evaluator.nbParts // 3)
    sequences = evaluator._decode_sequences(chromosomes, machine_idx)
    contexts = evaluator._init_batch_contexts(sequences, machine_idx, num_solutions,
                                               mach_data, max_bins_per_sol)

    max_total_bins = num_solutions * max_bins_per_sol
    grid_states = torch.zeros((max_total_bins, H, W), dtype=torch.float32, device=evaluator.device)
    grid_ffts = torch.zeros((max_total_bins, H, W // 2 + 1), dtype=torch.complex64, device=evaluator.device)
    row_idx = torch.arange(H, device=evaluator.device).view(1, H, 1)
    col_idx = torch.arange(W, device=evaluator.device).view(1, 1, W)
    neg_inf = torch.tensor(-1e9, device=evaluator.device)

    # Run a few waves to build up state
    for wave in range(5):
        active = [c for c in contexts if not c.is_done and c.is_feasible]
        if not active:
            break
        evaluator._process_wave_true_batch(active, mach_data, grid_states, grid_ffts,
                                            row_idx, col_idx, neg_inf)

    # Now profile one wave in detail
    active = [c for c in contexts if not c.is_done and c.is_feasible]
    if not active:
        print("All contexts done after 5 waves")
        return

    # Gather context_info (Phase 1)
    context_info = []
    for ctx in active:
        if ctx.current_part_idx >= len(ctx.parts_sequence):
            ctx.is_done = True
            continue
        part_id = ctx.parts_sequence[ctx.current_part_idx]
        part_data = evaluator.parts[part_id]
        mach_part_data = mach_data.parts[part_id]
        shape0 = part_data.shapes[0]
        if (shape0[0] > H or shape0[1] > W) and (shape0[1] > H or shape0[0] > W):
            continue
        context_info.append((ctx, part_data, mach_part_data))

    n_contexts = len(context_info)
    print(f"\n=== Phase 3 Breakdown (wave with {n_contexts} contexts) ===")

    # Measure Phase 3a: Pass 1 collection
    t_vacancy_total = 0.0
    t_append_total = 0.0
    t_loop_total = 0.0
    n_vacancy_calls = 0
    n_appends = 0

    p1_n_tests = 0
    p1_lists = {k: [] for k in ['grid_indices', 'part_ffts', 'heights', 'widths',
                                  'bin_indices', 'shapes', 'bin_states', 'rotations',
                                  'ctx_indices', 'enclosure_lengths', 'bin_areas', 'part_areas']}
    ctx_first_valid_bin = [-1] * n_contexts

    t_phase3_start = time.perf_counter()

    for ctx_idx, (ctx, part_data, mach_part_data) in enumerate(context_info):
        for bin_idx, bin_state in enumerate(ctx.open_bins):
            t_loop_iter = time.perf_counter()

            if bin_state.area + part_data.area > ctx.bin_area:
                t_loop_total += time.perf_counter() - t_loop_iter
                continue

            rots_passing = []
            for rot in range(part_data.nrot):
                shape = part_data.shapes[rot]
                if shape[0] > H or shape[1] > W:
                    continue

                t_vac = time.perf_counter()
                vac_pass = check_vacancy_fit_simple(bin_state.vacancy_vector,
                                                    part_data.densities[rot])
                t_vacancy_total += time.perf_counter() - t_vac
                n_vacancy_calls += 1

                if vac_pass:
                    rots_passing.append((rot, shape))

            if rots_passing:
                ctx_first_valid_bin[ctx_idx] = bin_idx
                t_app = time.perf_counter()
                for rot, shape in rots_passing:
                    p1_lists['grid_indices'].append(bin_state.grid_state_idx)
                    p1_lists['part_ffts'].append(mach_part_data.ffts[rot])
                    p1_lists['heights'].append(shape[0])
                    p1_lists['widths'].append(shape[1])
                    p1_lists['bin_indices'].append(bin_idx)
                    p1_lists['shapes'].append(shape)
                    p1_lists['bin_states'].append(bin_state)
                    p1_lists['rotations'].append(rot)
                    p1_lists['ctx_indices'].append(ctx_idx)
                    p1_lists['enclosure_lengths'].append(bin_state.enclosure_box_length)
                    p1_lists['bin_areas'].append(bin_state.area)
                    p1_lists['part_areas'].append(part_data.area)
                    p1_n_tests += 1
                    n_appends += 1
                t_append_total += time.perf_counter() - t_app
                t_loop_total += time.perf_counter() - t_loop_iter
                break

            t_loop_total += time.perf_counter() - t_loop_iter

    t_phase3_total = time.perf_counter() - t_phase3_start

    print(f"  Total Phase 3a time: {t_phase3_total*1000:.2f}ms")
    print(f"    Vacancy checks: {t_vacancy_total*1000:.2f}ms ({n_vacancy_calls} calls, "
          f"{t_vacancy_total/max(n_vacancy_calls,1)*1e6:.1f}µs/call)")
    print(f"    List appends: {t_append_total*1000:.2f}ms ({n_appends} appends)")
    print(f"    Loop overhead: {(t_phase3_total - t_vacancy_total - t_append_total)*1000:.2f}ms")
    print(f"  Tests collected: {p1_n_tests}")

    # Measure Phase 5 tuple comparison
    print(f"\n=== Phase 5 Breakdown ===")

    # Simulate score components
    n_tests = p1_n_tests
    if n_tests == 0:
        print("  No tests to compare")
        return

    sc_bin_indices = np.random.randint(0, 5, n_tests).astype(np.float64)
    sc_densities = np.random.uniform(0.1, 0.9, n_tests)
    sc_rows = np.random.uniform(0, H, n_tests)
    sc_cols = np.random.uniform(0, W, n_tests)
    sc_valid = np.ones(n_tests, dtype=bool)
    test_ctx_indices = p1_lists['ctx_indices']

    # Method 1: Current tuple comparison
    t0 = time.perf_counter()
    for _ in range(100):
        best_ti_per_ctx = [-1] * n_contexts
        best_key_per_ctx = [None] * n_contexts
        for ti, ctx_idx in enumerate(test_ctx_indices):
            if not sc_valid[ti]:
                continue
            key = (-sc_bin_indices[ti], sc_densities[ti], sc_rows[ti], -sc_cols[ti])
            prev = best_key_per_ctx[ctx_idx]
            if prev is None or key > prev:
                best_key_per_ctx[ctx_idx] = key
                best_ti_per_ctx[ctx_idx] = ti
    t_tuple = (time.perf_counter() - t0) / 100

    # Method 2: Numpy-based lexicographic (hypothetical optimization)
    t0 = time.perf_counter()
    for _ in range(100):
        # Build composite score that preserves lexicographic order
        # Since bin_idx is integer 0-N, density in [0,1], row in [0,H], col in [0,W]
        # We can use: -bin_idx * 1e12 + density * 1e9 + row * 1e3 - col
        # With float64 this is exact for our ranges
        scores = (-sc_bin_indices * 1e12 + sc_densities * 1e9 + sc_rows * 1e3 - sc_cols)
        scores[~sc_valid] = -np.inf
        # Use numpy scatter-max equivalent
        best_ti_np = np.full(n_contexts, -1, dtype=np.int64)
        best_sc_np = np.full(n_contexts, -np.inf, dtype=np.float64)
        ctx_arr = np.array(test_ctx_indices, dtype=np.int64)
        for ti in range(n_tests):
            ci = ctx_arr[ti]
            if scores[ti] > best_sc_np[ci]:
                best_sc_np[ci] = scores[ti]
                best_ti_np[ci] = ti
    t_composite = (time.perf_counter() - t0) / 100

    print(f"  Tuple comparison: {t_tuple*1000:.2f}ms ({n_tests} tests, {n_contexts} contexts)")
    print(f"  Composite float64: {t_composite*1000:.2f}ms")
    print(f"  Ratio: {t_tuple/t_composite:.2f}x")


def profile_per_machine_time(evaluator, chromosomes):
    """Measure time spent per machine to see if machines are balanced."""
    print(f"\n=== Per-Machine Time ===")
    for m in range(evaluator.nbMachines):
        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            evaluator._process_machine_batch(chromosomes, m, len(chromosomes))
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        print(f"  Machine {m}: {np.mean(times)*1000:.1f}ms "
              f"(grid: {evaluator.machines[m].bin_length}x{evaluator.machines[m].bin_width})")


def profile_decode_overhead(evaluator, chromosomes):
    """Measure _decode_sequences overhead."""
    print(f"\n=== Decode Sequences Overhead ===")
    for m in range(evaluator.nbMachines):
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            seqs = evaluator._decode_sequences(chromosomes, m)
            times.append(time.perf_counter() - t0)
        avg_parts = np.mean([len(s) for s in seqs])
        print(f"  Machine {m}: {np.mean(times)*1000:.2f}ms "
              f"(avg {avg_parts:.0f} parts/solution)")


def count_waves_and_tests(evaluator, chromosomes):
    """Count total waves, tests per wave, etc."""
    print(f"\n=== Wave/Test Statistics ===")
    for m in range(evaluator.nbMachines):
        mach_data = evaluator.machines[m]
        H, W = mach_data.bin_length, mach_data.bin_width
        num_solutions = len(chromosomes)
        max_bins_per_sol = max(10, evaluator.nbParts // 3)
        sequences = evaluator._decode_sequences(chromosomes, m)

        avg_parts = np.mean([len(s) for s in sequences])
        max_parts = max(len(s) for s in sequences)

        print(f"  Machine {m} ({H}x{W}):")
        print(f"    Avg parts/solution: {avg_parts:.1f}, Max: {max_parts}")
        print(f"    Expected waves: ~{max_parts}")
        print(f"    Population: {num_solutions}")
        print(f"    Max bins/solution: {max_bins_per_sol}")


if __name__ == '__main__':
    nbParts = int(sys.argv[1])
    nbMachines = int(sys.argv[2])
    instNumber = int(sys.argv[3])
    backend_name = sys.argv[4] if len(sys.argv) > 4 else 'torch_gpu'

    print(f"Setting up P{nbParts}M{nbMachines}-{instNumber} with {backend_name}...")
    problem_data, nbParts, nbMachines, thresholds, instanceParts, _, collision_backend = setup_problem(
        nbParts, nbMachines, instNumber, backend_name)

    mult = 10
    num_individuals = mult * nbParts
    num_gene = 2 * nbParts

    evaluator = WaveBatchEvaluator(
        problem_data, nbParts, nbMachines, thresholds,
        instanceParts, collision_backend
    )

    # Generate population
    pop = np.random.uniform(0, 1, (num_individuals, num_gene)).astype(np.float32)
    chromosomes = np.array(pop)

    print(f"Population: {num_individuals} individuals, {num_gene} genes")

    # Warm up
    print("Warming up...")
    evaluator.evaluate_batch(chromosomes)

    # Profile overall generation time
    print("\n=== Overall Generation Time ===")
    gen_times = profile_generation_breakdown(evaluator, chromosomes, n_gens=5)
    print(f"  Mean: {np.mean(gen_times)*1000:.1f}ms, Std: {np.std(gen_times)*1000:.1f}ms")
    for i, t in enumerate(gen_times):
        print(f"  Gen {i}: {t*1000:.1f}ms")

    # Profile per-machine
    profile_per_machine_time(evaluator, chromosomes)

    # Profile decode
    profile_decode_overhead(evaluator, chromosomes)

    # Wave/test stats
    count_waves_and_tests(evaluator, chromosomes)

    # Phase 3 breakdown (only for small instances to avoid long output)
    if nbParts <= 100:
        profile_phase3_breakdown(evaluator, chromosomes, 0)

    print("\nDone.")
