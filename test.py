from indic_aug.align import Aligner
from indic_aug.embedding.tda import TDAugmentor

parallelAugmentor = TDAugmentor(
    'example/preprocessed/data.hi',
    'example/preprocessed/data.en',
    Aligner.load('example/aligner.pkl'),
    3000,
    'example/vocab'
)

for i in range(10):
    next(parallelAugmentor)