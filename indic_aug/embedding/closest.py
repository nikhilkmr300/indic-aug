import logging
from pathlib import Path
import random
import os

from ..globals import Augmentor, ERRORS
from ..utils import line_count, path2lang

def find_closest_in_vocab():
    pass

class ClosestEmbeddingAugmentor(Augmentor):
    def __init__(self, src_input_path, tgt_input_path, aligner, p, k, polyglot_dir=str(Path.home() / 'polyglot_data'), augment=True, random_state=1):
        if line_count(src_input_path) != line_count(tgt_input_path):
            raise RuntimeError(ERRORS['corpus_shape'])
        self.doc_count = line_count(src_input_path)

        self.src_lang = path2lang(src_input_path)
        self.tgt_lang = path2lang(tgt_input_path)

        self.aligner = aligner

        self.p = p
        self.k = k

    def __next__(self):
        pass

    def __iter__(self):
        return self

    def __len__(self):
        return self.doc_count