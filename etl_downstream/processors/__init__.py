"""Registry of downstream-dataset processors."""
from __future__ import annotations

from ..base_processor import DownstreamBaseProcessor
from .hwt import HWTProcessor
from .spacone_forearmbicep import SpaconeForearmBicepProcessor

PROCESSOR_REGISTRY: dict[str, type[DownstreamBaseProcessor]] = {
    "SpaconeForearmBicepProcessor": SpaconeForearmBicepProcessor,
    "HWTProcessor": HWTProcessor,
}

__all__ = [
    "DownstreamBaseProcessor",
    "HWTProcessor",
    "PROCESSOR_REGISTRY",
    "SpaconeForearmBicepProcessor",
]
