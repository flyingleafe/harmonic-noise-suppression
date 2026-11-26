"""MP-SENet, current official release (Lu, Ai, Ling; 2.26 M-parameter generator).

``generator`` — waveform-in/out generator (``build_mpsenet_official``);
``discriminator`` — PESQ metric discriminator; ``losses`` — every upstream
training-loss term plus :func:`~models.mpsenet_official.losses.mpsenet_losses`.
Distinct from the older hybrid port in ``models.mpsenet``.
"""

from models.mpsenet_official.discriminator import MetricDiscriminator, build_metric_discriminator
from models.mpsenet_official.generator import MPNet, MPSENetOfficial, build_mpsenet_official

__all__ = [
    "MPNet",
    "MPSENetOfficial",
    "MetricDiscriminator",
    "build_metric_discriminator",
    "build_mpsenet_official",
]
