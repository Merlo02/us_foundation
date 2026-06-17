"""Downstream task package for the ultrasound foundation model.

Exposes the encoder wrapper (tokenizer + encoder + pooling, decoder-less),
the swappable heads (classification / regression) and the Lightning task
module that wires everything together with freeze / fine-tune toggles
and optional pretrained-encoder loading.
"""

from .encoder_wrapper import UltrasonicEncoderWrapper
from .heads import (
    ChannelCrossAttentionHead,
    ChannelPosEnc,
    ClassificationHead,
    PosEncConcatHead,
    RegressionHead,
    build_head,
    normalize_head_type,
)
from .pooling import MeanPool, Pooling, build_pooling
from .us_downstream import UltrasonicDownstream

__all__ = [
    "ChannelCrossAttentionHead",
    "ChannelPosEnc",
    "ClassificationHead",
    "MeanPool",
    "PosEncConcatHead",
    "Pooling",
    "RegressionHead",
    "UltrasonicDownstream",
    "UltrasonicEncoderWrapper",
    "build_head",
    "build_pooling",
    "normalize_head_type",
]
