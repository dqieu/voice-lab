"""Deterministic, model-free speech activity and voiced/unvoiced segmentation."""

from .dsp import Config
from .recommended import MeasurementConfig, analyze_recommended
from .representation import load_packet, save_packet, decode_packet

# The toolkit has one analysis pipeline, including six-stream reconstruction.
analyze = analyze_recommended

__all__ = ["Config", "analyze", "MeasurementConfig", "analyze_recommended", "load_packet", "save_packet", "decode_packet"]

__version__ = "0.1.0"
