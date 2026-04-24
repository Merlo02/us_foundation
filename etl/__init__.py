"""ETL pipeline for ultrasound A-mode signal preprocessing."""

from .config import DatasetConfig, ETLConfig
from .processors import (
    PROCESSOR_REGISTRY,
    BaseDatasetProcessor,
    BraushContractionProcessor,
    BraushFatigueProcessor,
    GiordanoHeartrateProcessor,
    GRAWUSProcessor,
    HWCProcessor,
    LateralGastrocnemiusVerasonicsProcessor,
    PICMUSCarotidCrossProcessor,
    PICMUSCarotidLongProcessor,
    PICMUSInVivoHeartProcessor,
    RawSample,
)

__all__ = [
    "BaseDatasetProcessor",
    "BraushContractionProcessor",
    "BraushFatigueProcessor",
    "DatasetConfig",
    "ETLConfig",
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
