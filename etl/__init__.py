"""ETL pipeline for ultrasound A-mode signal preprocessing (pretraining).

The labeled / multi-channel downstream ETL lives in :mod:`etl_downstream`.
"""

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
