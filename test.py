from indic_aug.depparse import DepParseAugmentor
from indic_aug.basic import NoisingAugmentor, DropoutAugmentor, SynonymAugmentor

# Dependency parsing demo
# aug = DepParseAugmentor('example/data/preprocessed/dev.en', 'example/data/preprocessed/dev.hi', 'blank', 0.1)
# print(next(aug))

# # Blanking demo
# aug = NoisingAugmentor('example/data/preprocessed/dev.en', 'example/data/preprocessed/dev.hi', 'kneser_ney', 0.1)
# print(next(aug))

# # Dropout demo
# aug = DropoutAugmentor('example/data/preprocessed/dev.en', 'example/data/preprocessed/dev.hi', 0.3)
# print(next(aug))

# Synonym demo
# aug = SynonymAugmentor('example/data/preprocessed/dev.en', 'example/data/preprocessed/dev.hi', 0.5, random_state=1)
# print(next(aug))