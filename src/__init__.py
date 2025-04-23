__version__ = "0.1.0"
__author__ = "Your Name <your.email@example.com>"
__license__ = "MIT"

# Package-level imports
from .grid_approach_test import (
    fast_factorial,
    Assign,
    generate_partitions,
    multi_coeff,
    spectral_embed,
    k_near_neighbor,
    preprocessing,
    cal_posterior_prob,
    first_order_derivative
)

# Numba configuration (optional)
from numba import config
config.FASTMATH = True  # Enable fastmath by default for Numba JIT

# Define public API
__all__ = [
    'fast_factorial',
    'Assign',
    'generate_partitions',
    'multi_coeff',
    'spectral_embed',
    'k_near_neighbor',
    'preprocessing',
    'cal_posterior_prob',
    'first_order_derivative',
    'LOOKUP_TABLE'
]

# Initialize logging (optional)
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())