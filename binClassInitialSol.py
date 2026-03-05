import numpy as np
import time
from collision_backend import create_collision_backend

class BuildingPlate:
    def __init__(self, width, length, collision_backend=None):
        #### NESTING CHARACTERISTICS ####
        self.width = width
        self.length = length
        self.enclosure_box_length = 0
        self.enclosure_box_width = 0
        self.area = 0
        # Track enclosure box bounds incrementally (avoid full grid scans)
        self.min_occupied_row = length  # No rows occupied yet
        self.max_occupied_row = -1
        # Initialize the grid with zeros (no parts placed yet)
        self.grid = np.zeros((length, width), dtype=np.uint8)
        # Initialize Vacancy Vector (VV) with maximum possible vacancy
        self.vacancy_vector = np.zeros(length, dtype=int) + width

        #### SCHEDULING CHARACTERISTICS ####
        self.processingTime = 0
        self.processingTimeHeight = 0
        self.partsAssigned = []

        self.resultFFT = 0
        self.collision_backend = collision_backend or create_collision_backend("torch_gpu")
        self.grid_state = self.collision_backend.create_grid_state(length, width)
        
        # Pre-allocate reusable buffers for vacancy vector updates (avoid allocations per insert)
        self._padded_buffer = np.ones((length, width + 2), dtype=np.uint8)
        self._max_zeros_buffer = np.zeros(length, dtype=np.int32)


    def save_plate_to_file(self, filename):
        with open(filename, 'w') as file:
            for row in self.grid:
                file.write(' '.join(f'{val:2d}' for val in row) + '\n')

    #@profile
    def can_insert(self, part, machPart):
        """Check if part can be inserted. Uses PartData and MachinePartData dataclasses."""
        result = False
        best_pixel, best_rotation, packingDensity = None, 0, 0 # Initialize packing density at zero  
        potentialArea = (self.area + part.area)

        # Get tensor from current binary grid
        startTim = time.time()

        feasible_rotations = []
        feasible_shapes = []
        feasible_ffts = []
        
        # Pre-extract values to avoid repeated attribute lookups in loop
        nrot = part.nrot
        vacancy = self.vacancy_vector
        part_shapes = part.shapes
        part_densities = part.densities
        machPart_ffts = machPart.ffts

        for currRot in range(nrot):
            shape = part_shapes[currRot]
            dens = part_densities[currRot]
            
            subarrays = np.lib.stride_tricks.sliding_window_view(vacancy, shape[0])
            binaryResult = np.any(np.all(subarrays >= dens, axis=1))
            
            if binaryResult:
                feasible_rotations.append(currRot)
                feasible_shapes.append(shape)
                feasible_ffts.append(machPart_ffts[currRot])

        batch_results = self.collision_backend.find_bottom_left_zero_batch(
            self.grid,
            feasible_ffts,
            feasible_shapes,
            grid_state=self.grid_state,
        )
        for i, currRot in enumerate(feasible_rotations):
            feasible, smallest_col_with_zero, largest_row_with_zero_real_value = batch_results[i]
            if feasible:
                result = True
                largest_row_with_zero = largest_row_with_zero_real_value - part_shapes[currRot][0] + 1

                newLength = max(self.enclosure_box_length, self.length - largest_row_with_zero)
                newPackingDensity = potentialArea/(newLength*self.width)

                if  newPackingDensity > packingDensity:
                    best_pixel, best_rotation, packingDensity = [smallest_col_with_zero,largest_row_with_zero_real_value], currRot, newPackingDensity

                elif newPackingDensity == packingDensity and largest_row_with_zero_real_value > best_pixel[1]:
                    best_pixel, best_rotation, packingDensity = [smallest_col_with_zero,largest_row_with_zero_real_value], currRot, newPackingDensity
                
                elif newPackingDensity == packingDensity and largest_row_with_zero_real_value == best_pixel[1] and smallest_col_with_zero < best_pixel[0]:
                    best_pixel, best_rotation, packingDensity = [smallest_col_with_zero,largest_row_with_zero_real_value], currRot, newPackingDensity

        
        if result == True:
            return result, best_pixel, best_rotation

        return result, 0, 0
    
    
    def calculate_enclosure_box_length(self):
        # O(1) lookup using tracked bounds instead of O(length × width) scan
        if self.min_occupied_row > self.max_occupied_row:
            self.enclosure_box_length = 0
        else:
            self.enclosure_box_length = self.length - self.min_occupied_row

    
    def insert(self, x, y, partMatrix, shapes, partArea):
        self.area += partArea
        
        y_start = y - shapes[0] + 1
        y_end = y + 1
        
        # Use slicing to insert the binary part matrix (cast to uint8 to match grid dtype)
        self.grid[y_start:y_end, x:x + shapes[1]] += partMatrix.astype(np.uint8)
        self.collision_backend.update_grid_region(self.grid_state, x, y, partMatrix, shapes)
        
        # Update enclosure box bounds incrementally - O(1)
        self.min_occupied_row = min(self.min_occupied_row, y_start)
        self.max_occupied_row = max(self.max_occupied_row, y)

        # Update vacancy vector using pre-allocated buffers (reduces allocations)
        num_rows = y_end - y_start
        
        # Use pre-allocated padded buffer (only the needed rows)
        padded = self._padded_buffer[:num_rows, :]
        padded[:, 0] = 1  # Left pad
        padded[:, -1] = 1  # Right pad
        padded[:, 1:-1] = self.grid[y_start:y_end, :]
        
        # Compute differences
        diffs = np.diff(padded.astype(np.int8), axis=1)
        
        # Identify start and end of zero runs
        start_indices = np.where(diffs == -1)
        end_indices = np.where(diffs == 1)
        
        # Compute lengths of zero runs
        run_lengths = end_indices[1] - start_indices[1]
        
        # Use pre-allocated buffer for max computation
        max_zeros = self._max_zeros_buffer[:num_rows]
        max_zeros.fill(0)
        np.maximum.at(max_zeros, start_indices[0], run_lengths)
        
        # Update the vacancy vector
        self.vacancy_vector[y_start:y_end] = max_zeros
            
