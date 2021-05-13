from pathlib import Path
import textwrap

import numpy as np

from ..globals import Augmentor, ERRORS
from ..vocab import Vocab
from ..utils import line_count, path2lang, cyclic_read
from ..log import logger, NUM_LOGGER_DASHES
from .common import fetch_embeddings

# The max number of closest neighbors for `find_closest_in_vocab` to search for
# an in-vocabulary neighbor. If no in-vocabulary neighbor is found among the
# MAX_CLOSEST number of closest neighbors, returns the original word.
MAX_NEIGHBORS = 30

def find_closest_in_vocab(word, embeddings, vocab):
    try:
        embeddings.nearest_neighbors(word, MAX_NEIGHBORS)
    except KeyError:
        # Word not in embeddings.
        logger.info(f'\t\tWord \'{word}\' does not have corresponding embedding.')
        return word

    for neighbor in embeddings.nearest_neighbors(word, MAX_NEIGHBORS):
        if neighbor in vocab:
            logger.info(f'\t\tReplaced source word \'{word}\' with its closest in-vocab neighbor \'{neighbor}\'.')
            return neighbor

    logger.info(f'\t\tNo in-vocab neighbor found for \'{word}\' among {MAX_NEIGHBORS} closest neighbors.')

    return word

class ClosestEmbeddingAugmentor(Augmentor):
    """Class to perform parallel augmentation by probabilistically replacing
    source words with their closest neighbors measured using cosine similarity
    between their word vector representations. Aligned target words are suitably
    replaced to preserve semantics.
    """

    def __init__(self, src_input_path, tgt_input_path, aligner, p, vocab_dir, polyglot_dir=None, augment=True):
        """Constructor method.

        :param src_input_path: Path to aligned source corpus.
        :type src_input_path: str
        :param tgt_input_path: Path to aligned target corpus, corresponding to
            the above source corpus.
        :type tgt_input_path: str
        :param aligner: Aligner to perform alignment between source and target
            sentences.
        :type aligner: ``indic_aug.align.Aligner``
        :param p: Probability with which to replace each word. Values are
            clipped to range [0, 1].
        :type p: float
        :param vocab_dir: As described in the docstring for
            ``indic_aug.vocab.Vocab``.
        :type vocab_dir: str
        :param polyglot_dir: Path to directory containing ``polyglot``
            embeddings. By default, downloads to '~/polyglot_data'.
            Must have same tree structure as used by 
            ``polyglot.downloader.dowloader``. For example, if ``polyglot_dir`` 
            is named 'polyglot_data', then directory 'polyglot_data' might look 
            like the following:

            .. code-block:: text

                polyglot_data
                └── embeddings2
                    └── en
                        └── embeddings_pkl.tar.bz2
                    └── hi
                        └── embeddings_pkl.tar.bz2
                    └── mr
                        └── embeddings_pkl.tar.bz2
                    └── ta
                        └── embeddings_pkl.tar.bz2
                    └── te
                        └── embeddings_pkl.tar.bz2

        :type polyglot_dir: str
        """
        if line_count(src_input_path) != line_count(tgt_input_path):
            raise RuntimeError(ERRORS['corpus_shape'])
        self.doc_count = line_count(src_input_path)

        if polyglot_dir is None:
            polyglot_dir = str(Path.home() / 'polyglot_data')

        self.src_lang = path2lang(src_input_path)
        self.tgt_lang = path2lang(tgt_input_path)

        self.aligner = aligner

        self.src_embeddings = fetch_embeddings(self.src_lang, polyglot_dir)

        # No need for tgt_vocab as `Aligner.src2tgt` will always give an
        # in-vocab target word. NOTE: This will be the case when the aligner has
        # been trained on the preprocessed corpus (i.e., with OOV words replaced
        # by UNK_TOKEN), so make sure you do that first.
        self.src_vocab = Vocab.load_vocab(vocab_dir, self.src_lang)

        self.augment = augment

        if self.augment:
            # If augment is True, can perform arbitrary number of augmentations
            # by cycling through all the sentences in the corpus repeatedly.
            self.src_input_file = cyclic_read(src_input_path)
            self.tgt_input_file = cyclic_read(tgt_input_path)
        else:
            # Else does one pass through the corpus and stops.
            self.src_input_file = open(src_input_path)
            self.tgt_input_file = open(tgt_input_path)

        self.p = p

        logger.info(textwrap.dedent(f'\
            ClosestEmbeddingAugmentor\n\
            \tdoc_count={self.doc_count}\n\
            \tsrc_input_path={src_input_path}\n\
            \ttgt_input_path={tgt_input_path}\n\
            \tsrc_lang={self.src_lang}\n\
            \ttgt_lang={self.tgt_lang}\n\
            \tp={self.p}\n\
            \tvocab_dir={vocab_dir}\n\
            Note: Words are 0-indexed.'
        ))
        logger.info('-' * NUM_LOGGER_DASHES)

    def __next__(self):
        # Returning original sentences as they are if self.augment is False.
        if not self.augment:
            return next(self.src_input_file).rstrip('\n'), next(self.tgt_input_file).rstrip('\n')

        src_doc = next(self.src_input_file).rstrip('\n').split(' ')
        tgt_doc = next(self.tgt_input_file).rstrip('\n').split(' ')

        logger.info(f'src_doc=\'{" ".join(src_doc)}\'')
        logger.info(f'tgt_doc=\'{" ".join(tgt_doc)}\'\n')

        augmented_src_doc = list()
        augmented_tgt_doc = tgt_doc.copy()      # .copy() to avoid aliasing.

        # Generating alignment between source and target documents.
        alignment = self.aligner.align(' '.join(src_doc), ' '.join(tgt_doc)).alignment

        for src_idx, src_word in enumerate(src_doc):
            if np.random.binomial(1, self.p):
                logger.info(f'\tSampled word \'{src_word}\' at index {src_idx} for replacement.')

                # Replace
                src_replacement = find_closest_in_vocab(src_word, self.src_embeddings, self.src_vocab)
                augmented_src_doc.append(src_word)

                # Finding corresponding word to s_{i} in target sentence.
                tgt_idxs = [t for t, s in alignment if s == src_idx]

                logger.info(f'\t\talignment(\'{src_word}\')={[(tgt_idx, tgt_doc[tgt_idx]) for tgt_idx in tgt_idxs]}.')

                # Replacing words at tgt_idxs with translation of
                # src_replacement.
                tgt_replacement = self.aligner.src2tgt(src_replacement)
                for tgt_idx in tgt_idxs:
                    augmented_tgt_doc[tgt_idx] = tgt_replacement
                    logger.info(f'\t\t\tReplaced target word \'{tgt_doc[tgt_idx]}\' with \'{tgt_replacement}\', which is the translation/alignment of \'{src_replacement}\'.')

            else:
                augmented_src_doc.append(src_word)

        augmented_src_doc = ' '.join(augmented_src_doc)
        augmented_tgt_doc = ' '.join(augmented_tgt_doc)

        logger.info(f'\naugmented_src_doc=\'{augmented_src_doc}\'')
        logger.info(f'augmented_tgt_doc=\'{augmented_tgt_doc}\'')
        logger.info('-' * NUM_LOGGER_DASHES)

        return augmented_src_doc, augmented_tgt_doc

    def __iter__(self):
        return self

    def __len__(self):
        return self.doc_count