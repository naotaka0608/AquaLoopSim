from .config import GRID_RES, DEFAULT_NUM_PARTICLES
from .solver_cpu import FluidSolverCPU

try:
    from .solver_gpu import FluidSolverGPU, HAS_CUPY
except ImportError:
    FluidSolverGPU = None
    HAS_CUPY = False

def FluidSolver(res_x=GRID_RES[0], res_y=GRID_RES[1], res_z=GRID_RES[2], num_particles=DEFAULT_NUM_PARTICLES, backend='cpu'):
    """
    Factory function to return a FluidSolver instance.
    backend: 'cpu' or 'gpu'
    """
    if backend == 'gpu':
        if HAS_CUPY and FluidSolverGPU is not None:
             print("Using GPU Solver (CuPy)")
             return FluidSolverGPU(res_x, res_y, res_z, num_particles)
        else:
             print("Warning: GPU backend requested but CuPy is not available. Falling back to CPU.")
             return FluidSolverCPU(res_x, res_y, res_z, num_particles)
    else:
        print("Using CPU Solver (Numba)")
        return FluidSolverCPU(res_x, res_y, res_z, num_particles)
