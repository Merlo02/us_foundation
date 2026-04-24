"""Register all dataset-specific processors in :data:`PROCESSOR_REGISTRY`."""
from __future__ import annotations

from .base_processor import BaseDatasetProcessor, RawSample
from .braush_contraction_processor import BraushContractionProcessor
from .braush_fatigue_processor import BraushFatigueProcessor
from .giordano_heartrate_processor import GiordanoHeartrateProcessor
from .grawus_processor import GRAWUSProcessor
from .hwc_processor import HWCProcessor
from .lateral_gastrocnemius_verasonics_processor import (
    LateralGastrocnemiusVerasonicsProcessor,
)
from .picmus_carotid_cross_processor import PICMUSCarotidCrossProcessor
from .picmus_carotid_long_processor import PICMUSCarotidLongProcessor
from .picmus_in_vivo_heart_processor import PICMUSInVivoHeartProcessor

PROCESSOR_REGISTRY: dict[str, type[BaseDatasetProcessor]] = {
    "BraushContractionProcessor": BraushContractionProcessor,
    "BraushFatigueProcessor": BraushFatigueProcessor,
    "GiordanoHeartrateProcessor": GiordanoHeartrateProcessor,
    "HWCProcessor": HWCProcessor,
    "GRAWUSProcessor": GRAWUSProcessor,
    "LateralGastrocnemiusVerasonicsProcessor": LateralGastrocnemiusVerasonicsProcessor,
    "PICMUSCarotidCrossProcessor": PICMUSCarotidCrossProcessor,
    "PICMUSCarotidLongProcessor": PICMUSCarotidLongProcessor,
    "PICMUSInVivoHeartProcessor": PICMUSInVivoHeartProcessor,
}

__all__ = [
    "BaseDatasetProcessor",
    "BraushContractionProcessor",
    "BraushFatigueProcessor",
    "GiordanoHeartrateProcessor",
    "GRAWUSProcessor",
    "HWCProcessor",
    "LateralGastrocnemiusVerasonicsProcessor",
    "PICMUSCarotidCrossProcessor",
    "PICMUSCarotidLongProcessor",
    "PICMUSInVivoHeartProcessor",
    "PROCESSOR_REGISTRY",
    "RawSample",
]
