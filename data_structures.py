"""
Data structures for FFT Placement optimization.

This module defines dataclasses to replace nested dictionaries with string keys,
providing faster attribute access and better code clarity.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class PartData:
    """
    Data for a single part (geometry, rotations, etc.)
    Independent of machine - computed once per part.
    """
    id: int
    area: float
    nrot: int
    rotations: List[np.ndarray]      # [rot0, rot1, rot2, rot3] - contiguous arrays
    shapes: List[Tuple[int, int]]    # [(h0, w0), (h1, w1), ...] for each rotation
    densities: List[np.ndarray]      # Max consecutive 1s per row for each rotation
    best_rotation: int               # Index of rotation with minimum height
    rotations_gpu: Optional[List[Any]] = None  # Pre-transferred GPU tensors (avoids CPU->GPU per insert)
    
    @property
    def lengths(self) -> List[int]:
        """Height of each rotation."""
        return [s[0] for s in self.shapes]


@dataclass
class MachinePartData:
    """
    Machine-specific data for a part.
    Depends on both part and machine (FFTs, processing times).
    """
    ffts: List[Any]        # Pre-computed FFTs for each rotation (torch tensors or numpy arrays)
    proc_time: float       # Processing time on this machine
    proc_time_height: float  # Height-based processing time


@dataclass
class MachineData:
    """
    Data for a single machine (dimensions, setup time, part-specific data).
    """
    bin_length: int
    bin_width: int
    bin_area: int
    setup_time: float
    parts: Dict[int, MachinePartData] = field(default_factory=dict)  # part_id -> MachinePartData


@dataclass
class ProblemData:
    """
    Container for all problem data.
    Provides fast integer-indexed access to parts and machines.
    """
    parts: Dict[int, PartData]           # part_id -> PartData
    machines: List[MachineData]          # machine_index -> MachineData
    instance_parts: np.ndarray           # Original instance parts array
    instance_parts_unique: np.ndarray    # Unique part IDs
    
    def get_part(self, part_id: int) -> PartData:
        """Get part data by ID."""
        return self.parts[part_id]
    
    def get_machine(self, machine_idx: int) -> MachineData:
        """Get machine data by index."""
        return self.machines[machine_idx]
    
    def get_machine_part(self, machine_idx: int, part_id: int) -> MachinePartData:
        """Get machine-specific part data."""
        return self.machines[machine_idx].parts[part_id]
