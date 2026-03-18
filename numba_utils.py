"""
Numba JIT-compiled functions for performance-critical operations.

These functions accelerate the vacancy checking in can_insert() which is called 100K+ times.
"""

import numpy as np
from numba import jit, prange
from numba.typed import List as NumbaList


@jit(nopython=True, cache=True, fastmath=True)
def check_vacancy_fit_single(vacancy: np.ndarray, density: np.ndarray, shape_height: int) -> bool:
    """
    Check if a part with given density profile can fit in the vacancy vector.
    
    This is the JIT-compiled equivalent of:
        subarrays = np.lib.stride_tricks.sliding_window_view(vacancy, shape_height)
        return np.any(np.all(subarrays >= density, axis=1))
    
    Args:
        vacancy: 1D array of max contiguous zeros per row (int64)
        density: 1D array of max consecutive 1s per row for the part (int32)
        shape_height: Height of the part (same as len(density))
    
    Returns:
        True if part can potentially fit, False otherwise
    """
    n = len(vacancy)
    window_size = shape_height
    
    if window_size > n:
        return False
    
    # Slide window over vacancy vector
    # Check if any window position has all vacancy[i] >= density[i]
    for start in range(n - window_size + 1):
        fits = True
        for i in range(window_size):
            if vacancy[start + i] < density[i]:
                fits = False
                break
        if fits:
            return True
    
    return False


@jit(nopython=True, cache=True, fastmath=True)
def check_rotations_feasibility(
    vacancy: np.ndarray,
    densities_flat: np.ndarray,
    density_offsets: np.ndarray,
    shapes_heights: np.ndarray,
    shapes_widths: np.ndarray,
    bin_length: int,
    bin_width: int,
    nrot: int
) -> np.ndarray:
    """
    Check all rotations for feasibility in a single JIT-compiled function.
    
    Args:
        vacancy: 1D vacancy vector (int64)
        densities_flat: Flattened array of all density arrays concatenated
        density_offsets: Start index of each rotation's density in densities_flat
        shapes_heights: Array of heights for each rotation
        shapes_widths: Array of widths for each rotation
        bin_length: Bin length
        bin_width: Bin width
        nrot: Number of rotations
    
    Returns:
        Boolean array indicating feasibility of each rotation
    """
    results = np.zeros(nrot, dtype=np.bool_)
    
    for rot in range(nrot):
        h = shapes_heights[rot]
        w = shapes_widths[rot]
        
        # Skip rotations that don't fit bin dimensions
        if h > bin_length or w > bin_width:
            continue
        
        # Get this rotation's density array
        start_idx = density_offsets[rot]
        end_idx = density_offsets[rot + 1] if rot + 1 < len(density_offsets) else len(densities_flat)
        density = densities_flat[start_idx:end_idx]
        
        # Check if this rotation can fit
        results[rot] = check_vacancy_fit_single(vacancy, density, h)
    
    return results


def prepare_rotation_data_for_jit(part_densities, part_shapes):
    """
    Prepare rotation data in a format suitable for Numba JIT functions.
    
    This should be called once per part (during problem setup) rather than
    per can_insert call.
    
    Args:
        part_densities: List of density arrays for each rotation
        part_shapes: List of (height, width) tuples
    
    Returns:
        Tuple of (densities_flat, density_offsets, shapes_heights, shapes_widths)
    """
    # Flatten all densities into a single array with offset tracking
    densities_flat = np.concatenate([d.astype(np.int32) for d in part_densities])
    
    # Track where each rotation's density starts
    density_offsets = np.zeros(len(part_densities) + 1, dtype=np.int32)
    offset = 0
    for i, d in enumerate(part_densities):
        density_offsets[i] = offset
        offset += len(d)
    density_offsets[-1] = offset
    
    # Extract shapes as separate arrays
    shapes_heights = np.array([s[0] for s in part_shapes], dtype=np.int32)
    shapes_widths = np.array([s[1] for s in part_shapes], dtype=np.int32)
    
    return densities_flat, density_offsets, shapes_heights, shapes_widths


# Simple version that works with existing data structures (no pre-processing needed)
@jit(nopython=True, cache=True, fastmath=True)
def check_vacancy_fit_simple(vacancy: np.ndarray, density: np.ndarray) -> bool:
    """
    Simplified vacancy check - works directly with density array.
    
    Args:
        vacancy: 1D int64 vacancy vector
        density: 1D int32 density array for one rotation
    
    Returns:
        True if part can fit, False otherwise
    """
    n = len(vacancy)
    window_size = len(density)
    
    if window_size > n:
        return False
    
    for start in range(n - window_size + 1):
        fits = True
        for i in range(window_size):
            if vacancy[start + i] < density[i]:
                fits = False
                break
        if fits:
            return True
    
    return False


@jit(nopython=True, cache=True)
def update_vacancy_vector_rows(vacancy_vector: np.ndarray, grid_rows: np.ndarray, y_start: int) -> None:
    """
    Update vacancy vector for modified rows after part insertion.
    
    Computes max consecutive zeros per row in a single pass.
    Much faster than numpy diff + where + maximum.at approach.
    
    Args:
        vacancy_vector: 1D vacancy vector to update in-place
        grid_rows: 2D array of grid rows that were modified (already sliced)
        y_start: Starting row index in vacancy_vector
    """
    num_rows = grid_rows.shape[0]
    width = grid_rows.shape[1]
    
    for i in range(num_rows):
        max_zeros = 0
        current_zeros = 0
        
        for j in range(width):
            if grid_rows[i, j] == 0:
                current_zeros += 1
                if current_zeros > max_zeros:
                    max_zeros = current_zeros
            else:
                current_zeros = 0
        
        vacancy_vector[y_start + i] = max_zeros
