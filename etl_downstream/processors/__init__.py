"""Registry of downstream-dataset processors."""
from __future__ import annotations

from ..base_processor import DownstreamBaseProcessor
from .spacone_forearmbicep import SpaconeForearmBicepProcessor

PROCESSOR_REGISTRY: dict[str, type[DownstreamBaseProcessor]] = {
    "SpaconeForearmBicepProcessor": SpaconeForearmBicepProcessor,
}

__all__ = [
    "DownstreamBaseProcessor",
    "PROCESSOR_REGISTRY",
    "SpaconeForearmBicepProcessor",
]
