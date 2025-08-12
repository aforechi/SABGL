# __init__.py

"""
VGRAM: A Python Wrapper for the V-GRAM Weightless Neural Network Library
"""

__version__ = "1.1.0"

# Expose the main user-facing classes and functions
from .vgram_core import VGRAM
from .vgram_synapse import (
    Connection,
    ConnectionInput,
    ConnectionRandom,
    ConnectionGaussian,
    ConnectionLogPolar,
)
from . import vgram_output as NetworkOutput
from . import vgram_image as ImageProcProxy

__all__ = [
    "VGRAM",
    "Connection",
    "ConnectionInput",
    "ConnectionRandom",
    "ConnectionGaussian",
    "ConnectionLogPolar",
    "NetworkOutput",
    "ImageProcProxy",
]