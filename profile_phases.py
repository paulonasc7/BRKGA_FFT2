#!/usr/bin/env python3
"""
Accurate per-phase wall-clock profiling of wave_batch_evaluator.
Uses torch.cuda.synchronize() before each timer so GPU work is
included in the measured interval — no cProfile distortion.

Run: python profile_phases.py 50 2 0 torch_gpu 3
     (nbParts nbMachines instNumber backend n_generations)
"""
import sys, time, math, os, pickle
import numpy as np
import torch
import pandas as pd
from collections import defaultdict

from collision_backend import create_collision_backend
from data_structures import PartData, MachinePartData, MachineData, ProblemData
from wave_batch_evaluator import WaveBatchEvaluator, BinState
from numba_utils import check_vacancy_fit_simple, update_vacancy_vector_rows

torch.set_num_threads(1)
torch.set_grad_enabled(False)

# ── Args ──────────────────────────────────────────────────────────────────────
nbParts      = int(sys.argv[1]) if len(sys.argv) > 1 else 50
nbMachines   = int(sys.argv[2]) if len(sys.argv) > 2 else 2
instNumber   = int(sys.argv[3]) if len(sys.argv) > 3 else 0
backend_name = sys.argv[4]      if len(sys.argv) > 4 else 'torch_gpu'
n_gen        = int(sys.argv[5]) if len(sys.argv) > 5 else 3
NUM_INDIVIDUALS = 500

# ── Problem setup (mirrors BRKGA_alg3.py) ────────────────────────────────────
collision_backend = create_collision_backend(backend_name)

with open(f'data/Instances/P{nbParts}M{nbMachines}-{instNumber}.txt') as f:
    instanceParts = np.array([int(x) for x in f.read().split()])
instancePartsUnique = np.unique(instanceParts)

cache_path = 'data/PartsMachines/cached_specs.pkl'
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        jobSpecAll, machSpec, area, polRotations = pickle.load(f)
else:
    jobSpecAll   = pd.read_excel('data/PartsMachines/part-machine-information.xlsx', sheet_name='part',    header=0, index_col=0)
    machSpec     = pd.read_excel('data/PartsMachines/part-machine-information.xlsx', sheet_name='machine', header=0, index_col=0)
    area         = pd.read_excel('data/PartsMachines/polygon_areas.xlsx', header=0)["Area"].tolist()
    polRotations = pd.read_excel('data/PartsMachines/parts_rotations.xlsx', header=0)["rot"].tolist()
    with open(cache_path, 'wb') as f:
        pickle.dump((jobSpecAll, machSpec, area, polRotations), f)

jobSpec = jobSpecAll.loc[instancePartsUnique]

parts_dict = {}
for part in instancePartsUnique:
    matrix = np.ascontiguousarray(np.load(f'data/partsMatrices/matrix_{part}.npy').astype(np.int32))
    nrot = 2 if np.array_equal(matrix, np.rot90(matrix, 2)) else 4
    rotations, shapes, densities = [], [], []
    for rot in range(nrot):
        r = np.ascontiguousarray(np.rot90(matrix, rot))
        rotations.append(r); shapes.append(r.shape)
        padded = np.pad(r, ((0,0),(1,1)), constant_values=0)
        diffs  = np.diff(padded.astype(np.int8), axis=1)
        si, ei = np.where(diffs==1), np.where(diffs==-1)
        rl = ei[1] - si[1]
        mx = np.zeros(r.shape[0], dtype=np.int32)
        if len(si[0]): np.maximum.at(mx, si[0], rl)
        densities.append(mx)
    best_rotation = int(np.argmin([s[0] for s in shapes]))
    rotations_gpu  = [collision_backend.prepare_rotation_tensor(r) for r in rotations] \
                     if hasattr(collision_backend, 'prepare_rotation_tensor') else None
    rotations_uint8 = [r.astype(np.uint8) for r in rotations]
    pd_ = PartData(id=part, area=area[part], nrot=nrot, rotations=rotations,
                   shapes=shapes, densities=densities, best_rotation=best_rotation,
                   rotations_gpu=rotations_gpu, rotations_uint8=rotations_uint8)
    pd_.prepare_jit_data()
    parts_dict[part] = pd_

machines_list = []
for m in range(nbMachines):
    bL, bW = machSpec['L(mm)'].iloc[m], machSpec['W(mm)'].iloc[m]
    st = machSpec['ST(s)'].iloc[m]
    mp = {}
    for part in instancePartsUnique:
        pd_ = parts_dict[part]
        ffts = [collision_backend.prepare_part_fft(pd_.rotations[r], bL, bW) for r in range(pd_.nrot)]
        pt  = jobSpec["volume(mm3)"].loc[part]*machSpec["VT(s/mm3)"].iloc[m] + \
              jobSpec["support(mm3)"].loc[part]*machSpec["SPT(s/mm3)"].iloc[m]
        pth = jobSpec["height(mm)"].loc[part]*machSpec["HT(s/mm3)"].iloc[m]
        mp[part] = MachinePartData(ffts=ffts, proc_time=pt, proc_time_height=pth)
    machines_list.append(MachineData(bin_length=bL, bin_width=bW, bin_area=bL*bW,
                                     setup_time=st, parts=mp))

problem_data = ProblemData(parts=parts_dict, machines=machines_list,
                           instance_parts=instanceParts,
                           instance_parts_unique=instancePartsUnique)
thresholds = [t/nbMachines for t in range(1, nbMachines)]

print(f"Setup done. Running {n_gen} generations × {NUM_INDIVIDUALS} individuals "
      f"(P{nbParts}M{nbMachines}-{instNumber}, {backend_name})")


# ── Timed evaluator ───────────────────────────────────────────────────────────
class TimedEvaluator(WaveBatchEvaluator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pt = defaultdict(float)   # phase → cumulative seconds
        self._n_waves = 0
        self._n_tests = 0

    def _s(self):
        torch.cuda.synchronize()
        return time.perf_counter()

    def _process_wave_true_batch(self, contexts, mach_data, grid_states, grid_ffts,
                                 row_idx, col_idx, neg_inf):
        if not contexts:
            return

        H, W = mach_data.bin_length, mach_data.bin_width
        self._n_waves += 1

        # ── Phase 1 ──────────────────────────────────────────────────────────
        t = self._s()
        context_info = []
        for ctx in contexts:
            if ctx.current_part_idx >= len(ctx.parts_sequence):
                ctx.is_done = True; continue
            part_id    = ctx.parts_sequence[ctx.current_part_idx]
            part_data  = self.parts[part_id]
            mach_part_data = mach_data.parts[part_id]
            shape0 = part_data.shapes[0]
            if (shape0[0]>H or shape0[1]>W) and (shape0[1]>H or shape0[0]>W):
                ctx.is_feasible = False; continue
            context_info.append((ctx, part_data, mach_part_data))
        if not context_info:
            return
        self._pt[1] += self._s() - t

        # ── Phase 2 ──────────────────────────────────────────────────────────
        t = self._s()
        invalid_idx, invalid_bs = [], []
        for ctx,_,_ in context_info:
            for bs in ctx.open_bins:
                if not bs.grid_fft_valid:
                    invalid_idx.append(bs.grid_state_idx); invalid_bs.append(bs)
        if invalid_idx:
            with torch.inference_mode():
                idx = torch.tensor(invalid_idx, device=self.device)
                grid_ffts[idx] = torch.fft.rfft2(grid_states[idx])
                for bs in invalid_bs: bs.grid_fft_valid = True
        self._pt[2] += self._s() - t

        # ── Phase 3a  Pass-1 collection (first valid bin per context) ────────
        t = self._s()
        n_contexts = len(context_info)
        ctx_first_valid_bin = [-1] * n_contexts
        p1_grid_indices, p1_part_ffts = [], []
        p1_heights, p1_widths = [], []
        p1_bin_indices, p1_shapes = [], []
        p1_bin_states, p1_rotations = [], []
        p1_ctx_indices, p1_enclosure_lengths = [], []
        p1_bin_areas, p1_part_areas = [], []
        p1_n_tests = 0
        for ctx_idx, (ctx, part_data, mach_part_data) in enumerate(context_info):
            for bin_idx, bin_state in enumerate(ctx.open_bins):
                if bin_state.area + part_data.area > ctx.bin_area:
                    continue
                rots_passing = []
                for rot in range(part_data.nrot):
                    shape = part_data.shapes[rot]
                    if shape[0]>H or shape[1]>W: continue
                    if check_vacancy_fit_simple(bin_state.vacancy_vector,
                                               part_data.densities[rot]):
                        rots_passing.append((rot, shape))
                if rots_passing:
                    ctx_first_valid_bin[ctx_idx] = bin_idx
                    for rot, shape in rots_passing:
                        p1_grid_indices.append(bin_state.grid_state_idx)
                        p1_part_ffts.append(mach_part_data.ffts[rot])
                        p1_heights.append(shape[0]); p1_widths.append(shape[1])
                        p1_bin_indices.append(bin_idx); p1_shapes.append(shape)
                        p1_bin_states.append(bin_state); p1_rotations.append(rot)
                        p1_ctx_indices.append(ctx_idx)
                        p1_enclosure_lengths.append(bin_state.enclosure_box_length)
                        p1_bin_areas.append(bin_state.area)
                        p1_part_areas.append(part_data.area)
                        p1_n_tests += 1
                    break
        self._pt['3a'] += self._s() - t

        # ── Phase 4a  IFFT for Pass 1 ─────────────────────────────────────────
        t = self._s()
        if p1_n_tests:
            p1_placement_results, p1_all_scores = self._batch_fft_all_tests(
                p1_n_tests, p1_grid_indices, p1_part_ffts, p1_heights, p1_widths,
                p1_bin_indices, p1_enclosure_lengths, p1_bin_areas, p1_part_areas,
                grid_ffts, H, W, row_idx, col_idx, neg_inf)
        else:
            p1_placement_results, p1_all_scores = [], np.array([], dtype=np.float32)
        self._pt['4a'] += self._s() - t

        # ── Phase 3b  Pass-2 collection (remaining bins for misses) ──────────
        t = self._s()
        ctx_p1_hit = [False] * n_contexts
        for ti, ctx_idx in enumerate(p1_ctx_indices):
            if p1_all_scores[ti] > -1e18:
                ctx_p1_hit[ctx_idx] = True
        p2_grid_indices, p2_part_ffts = [], []
        p2_heights, p2_widths = [], []
        p2_bin_indices, p2_shapes = [], []
        p2_bin_states, p2_rotations = [], []
        p2_ctx_indices, p2_enclosure_lengths = [], []
        p2_bin_areas, p2_part_areas = [], []
        p2_n_tests = 0
        for ctx_idx, (ctx, part_data, mach_part_data) in enumerate(context_info):
            if ctx_p1_hit[ctx_idx]:
                continue
            first_valid = ctx_first_valid_bin[ctx_idx]
            for bin_idx, bin_state in enumerate(ctx.open_bins):
                if bin_idx == first_valid:
                    continue
                if bin_state.area + part_data.area > ctx.bin_area:
                    continue
                for rot in range(part_data.nrot):
                    shape = part_data.shapes[rot]
                    if shape[0]>H or shape[1]>W: continue
                    if check_vacancy_fit_simple(bin_state.vacancy_vector,
                                               part_data.densities[rot]):
                        p2_grid_indices.append(bin_state.grid_state_idx)
                        p2_part_ffts.append(mach_part_data.ffts[rot])
                        p2_heights.append(shape[0]); p2_widths.append(shape[1])
                        p2_bin_indices.append(bin_idx); p2_shapes.append(shape)
                        p2_bin_states.append(bin_state); p2_rotations.append(rot)
                        p2_ctx_indices.append(ctx_idx)
                        p2_enclosure_lengths.append(bin_state.enclosure_box_length)
                        p2_bin_areas.append(bin_state.area)
                        p2_part_areas.append(part_data.area)
                        p2_n_tests += 1
        self._n_tests += p1_n_tests + p2_n_tests
        self._pt['3b'] += self._s() - t

        # ── Phase 4b  IFFT for Pass 2 ─────────────────────────────────────────
        t = self._s()
        if p2_n_tests:
            p2_placement_results, p2_all_scores = self._batch_fft_all_tests(
                p2_n_tests, p2_grid_indices, p2_part_ffts, p2_heights, p2_widths,
                p2_bin_indices, p2_enclosure_lengths, p2_bin_areas, p2_part_areas,
                grid_ffts, H, W, row_idx, col_idx, neg_inf)
        else:
            p2_placement_results, p2_all_scores = [], np.array([], dtype=np.float32)
        self._pt['4b'] += self._s() - t

        # Merge for Phase 5
        test_ctx_indices  = p1_ctx_indices  + p2_ctx_indices
        test_bin_states   = p1_bin_states   + p2_bin_states
        test_rotations    = p1_rotations    + p2_rotations
        test_shapes       = p1_shapes       + p2_shapes
        placement_results = p1_placement_results + p2_placement_results
        all_scores        = (np.concatenate([p1_all_scores, p2_all_scores])
                             if p2_n_tests else p1_all_scores)

        # ── Phase 5 ──────────────────────────────────────────────────────────
        t = self._s()
        n_contexts = len(context_info)
        best_ti_per_ctx = [-1] * n_contexts
        best_sc_per_ctx = np.full(n_contexts, -np.inf, dtype=np.float32)
        for ti, ctx_idx in enumerate(test_ctx_indices):
            sc = all_scores[ti]
            if sc > best_sc_per_ctx[ctx_idx]:
                best_sc_per_ctx[ctx_idx] = sc
                best_ti_per_ctx[ctx_idx] = ti
        contexts_needing_new_bin = []
        _placements = []
        for ctx_idx, (ctx, part_data, mach_part_data) in enumerate(context_info):
            ti = best_ti_per_ctx[ctx_idx]
            if ti == -1 or placement_results[ti] is None:
                contexts_needing_new_bin.append((ctx, part_data, mach_part_data)); continue
            col, row  = placement_results[ti]
            bin_state = test_bin_states[ti]
            rot       = test_rotations[ti]
            shape     = test_shapes[ti]
            bin_state.grid_fft_valid = False
            _placements.append((bin_state, col, row, rot, shape, part_data, mach_part_data))
            ctx.current_part_idx += 1
        if _placements:
            if self.flat_parts_gpu is not None:
                from wave_batch_evaluator import _cuda_batch_update
                _kernel_args = []
                for bin_state, col, row, rot, shape, pd_, _ in _placements:
                    y_start = row - shape[0] + 1
                    flat_offset, ph, pw = self.part_update_meta[(pd_.id, rot)]
                    _kernel_args.append(
                        (bin_state.grid_state_idx, y_start, col, flat_offset, ph, pw))
                _cuda_batch_update(grid_states, self.flat_parts_gpu, _kernel_args, H, W)
            else:
                for bin_state, col, row, rot, shape, pd_, _ in _placements:
                    y_start = row - shape[0] + 1
                    part_gpu = (pd_.rotations_gpu[rot]
                                if pd_.rotations_gpu and pd_.rotations_gpu[rot] is not None
                                else torch.as_tensor(pd_.rotations[rot],
                                                     dtype=torch.float32, device=self.device))
                    grid_states[bin_state.grid_state_idx, y_start:row+1,
                                col:col+shape[1]] += part_gpu
            for bin_state, col, row, rot, shape, pd_, mpd in _placements:
                y_start = row - shape[0] + 1
                bin_state.grid[y_start:row+1, col:col+shape[1]] += pd_.rotations_uint8[rot]
                update_vacancy_vector_rows(
                    bin_state.vacancy_vector, bin_state.grid[y_start:row+1, :], y_start)
                bin_state.area += pd_.area
                bin_state.min_occupied_row = min(bin_state.min_occupied_row, y_start)
                bin_state.max_occupied_row = max(bin_state.max_occupied_row, row)
                bin_state.enclosure_box_length = bin_state.bin_length - bin_state.min_occupied_row
                bin_state.proc_time += mpd.proc_time
                bin_state.proc_time_height = max(bin_state.proc_time_height, mpd.proc_time_height)
        self._pt[5] += self._s() - t

        # ── Phase 6 ──────────────────────────────────────────────────────────
        t = self._s()
        for ctx, part_data, mach_part_data in contexts_needing_new_bin:
            self._start_new_bin(ctx, part_data, mach_part_data, mach_data, grid_states)
            ctx.current_part_idx += 1
        self._pt[6] += self._s() - t

    def report(self):
        total = sum(self._pt.values())
        rows = [
            (1,    'Phase 1   gather context_info'),
            (2,    'Phase 2   batch grid FFTs'),
            ('3a', 'Phase 3a  P1 collect (first valid bin)'),
            ('4a', 'Phase 4a  P1 IFFT'),
            ('3b', 'Phase 3b  P2 collect (remaining bins)'),
            ('4b', 'Phase 4b  P2 IFFT'),
            (5,    'Phase 5   find best placements'),
            (6,    'Phase 6   open new bins'),
        ]
        t34a = self._pt['3a'] + self._pt['4a']
        t34b = self._pt['3b'] + self._pt['4b']
        print(f"\n{'='*70}")
        print(f"PHASE BREAKDOWN  ({self._n_waves} waves, {NUM_INDIVIDUALS} individuals)")
        print(f"{'='*70}")
        print(f"{'Phase':<41} {'Time(s)':>8} {'%':>7} {'ms/wave':>9}")
        print(f"{'-'*70}")
        for key, name in rows:
            t = self._pt[key]
            print(f"{name:<41} {t:>8.3f} {100*t/total:>6.1f}% "
                  f"{1000*t/max(self._n_waves,1):>8.2f}ms")
        print(f"{'-'*70}")
        print(f"  {'P1 subtotal (3a+4a)':<39} {t34a:>8.3f} {100*t34a/total:>6.1f}%")
        print(f"  {'P2 subtotal (3b+4b)':<39} {t34b:>8.3f} {100*t34b/total:>6.1f}%")
        print(f"{'-'*70}")
        print(f"{'TOTAL (wave fn only)':<41} {total:>8.3f} {'100.0%':>7}")
        print(f"\nAvg tests/wave : {self._n_tests/max(self._n_waves,1):.1f}")
        print(f"{'='*70}")


# ── Run ───────────────────────────────────────────────────────────────────────
evaluator = TimedEvaluator(problem_data, nbParts, nbMachines, thresholds,
                           instanceParts, collision_backend)

# Warm up (1 batch, not timed)
pop = np.random.uniform(0,1,(NUM_INDIVIDUALS, 2*nbParts)).astype(np.float32)
evaluator.evaluate_batch(pop)
evaluator._pt = defaultdict(float)
evaluator._n_waves = 0
evaluator._n_tests = 0

# Timed runs
gen_times = []
for g in range(n_gen):
    pop = np.random.uniform(0,1,(NUM_INDIVIDUALS, 2*nbParts)).astype(np.float32)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    evaluator.evaluate_batch(pop)
    torch.cuda.synchronize()
    gen_times.append(time.perf_counter() - t0)
    print(f"  gen {g}: {gen_times[-1]:.2f}s")

print(f"\nMean gen time: {np.mean(gen_times):.3f}s  "
      f"(std {np.std(gen_times):.3f}s)")
evaluator.report()
