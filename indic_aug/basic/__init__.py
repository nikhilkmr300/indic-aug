__all__ = [
    'NoisingAugmentor',
    'DropoutAugmentor',
    'SynonymAugmentor'
]

from .noising import NoisingAugmentor
from .dropout import DropoutAugmentor
from .synonym import SynonymAugmentor