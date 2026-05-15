"""Downstream task package for the ultrasound foundation model.

Exposes the encoder wrapper (tokenizer + encoder + pooling, decoder-less),
the swappable heads (classification / regression) and the Lightning task
module that wires everything together with freeze / fine-tune toggles
and optional pretrained-encoder loading.
"""

from .encoder_wrapper import UltrasonicEncoderWrapper
from .heads import ClassificationHead, RegressionHead, build_head
from .pooling import MeanPool, Pooling, build_pooling
from .us_downstream import UltrasonicDownstream

__all__ = [
    "ClassificationHead",
    "MeanPool",
    "Pooling",
    "RegressionHead",
    "UltrasonicDownstream",
    "UltrasonicEncoderWrapper",
    "build_head",
    "build_pooling",
]
