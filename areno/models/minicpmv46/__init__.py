"""MiniCPM-V 4.6 plugin.

Re-exports the adapter for the language-model trunk that lives under the HF
multimodal checkpoint (vision tower is not loaded here; the text trunk is
plugged into areno while the vision/projector pieces stay in the original
HF model when used end-to-end).
"""

from areno.models.minicpmv46.model import MiniCPMV46Adapter

__all__ = ["MiniCPMV46Adapter"]
