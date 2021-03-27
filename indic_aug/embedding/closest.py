import random

import numpy as np

from ..globals import ERRORS
from ..utils import cyclic_read, path2lang, line_count

def closest_embedding_aug(sent, embeddings, p, k):
    augmented_sent = list()
    sent = [word.strip() for word in sent.split(' ')]

    for word in sent:
        if word == '.' or word == '\u0964':
            # Not replacing fullstops.
            continue

        if np.random.binomial(1, p):
            # Randomly sampling from list of k nearest embeddings of word.
            try:
                sampled_word = random.sample(embeddings.nearest_neighbors(word, k), 1)[0]1
            except KeyError as e:
                sampled_word = word
            augmented_sent.append(sampled_word)
        else:
            augmented_sent.append(word)

    return ' '.join(augmented_sent)

class ClosestEmbeddingAugmentor:
    def __init__(self, src_input_path, tgt_input_path, p, k, augment=True, random_state=1):
        if line_count(src_input_path) != line_count(tgt_input_path):
            raise RuntimeError(ERRORS['corpus_shape'])
        self.doc_count = line_count(src_input_path)

        random.seed(random_state)               # synonym_aug uses random from standard library.
        np.random.seed(random_state)

        src_lang = path2lang(src_input_path)
        tgt_lang = path2lang(tgt_input_path)

        self.augment = augment

        if self.augment:
            # If augment is True, can perform arbitrary number of augmentations by cycling through all the sentences in the corpus repeatedly.
            self.src_input_file = cyclic_read(src_input_path)
            self.tgt_input_file = cyclic_read(tgt_input_path)
        else:
            # Else does one pass through the corpus and stops.
            self.src_input_file = open(src_input_path)
            self.tgt_input_file = open(tgt_input_path)
