from indic_aug.depparse import DepParseAugmentor
from indic_aug.basic.noising import NoisingAugmentor

# # Dependency parsing demo
# aug = DepParseAugmentor('example/data/preprocessed/dev.en', 'example/data/preprocessed/dev.hi', 'blank', 0.5)
# print(next(aug))

# Blanking demo
aug = NoisingAugmentor('example/data/preprocessed/dev.en', 'example/data/preprocessed/dev.hi', 'kneser_ney', 0.1)
print(next(aug))