import logging
import textwrap

logging.getLogger('numexpr.utils').setLevel(logging.WARN)

import numpy as np
import torch

np.random.seed(1)
torch.manual_seed(1)

from ..globals import Augmentor, SENTENCE_DELIMS, ERRORS, PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, BLANK_TOKEN
from ..lm import load as lm_load
from ..vocab import Vocab
from ..log import logger, NUM_LOGGER_DASHES
from ..utils import line_count, path2lang, cyclic_read

def extract_rare_words(vocab_dir, lang, word_count):
    """Wrapper around Vocab.read_bottomn_vocab. The difference is that this
    function removes words that we do not want replaced according to the
    translation data augmentation (TDA) algorithm (refer:
    :cite:t:`fadaee2017data`).

    :param vocab_dir: Path to directory containing ``sentencepiece`` \*.model
        and \*.vocab files, as described in ``indic_aug.vocab``.
    :type vocab_dir: str
    :param vocab_path: Path to \*.vocab file compatible with ``sentencepiece``,
        corresponding to model at ``model_path``.
    :type vocab_path: str
    :param word_count: Number of words to include in the set of targeted words.
        Note that the actual number of words returned might be slightly less 
        than the specified ``word_count`` as special tokens such as SOS, EOS, 
        UNK, etc. are removed.
    :type word_count: int

    :return: Output of ``Vocab.read_bottomn_vocab`` minus the special tokens we
        do not want replaced.
    :rtype: list
    """

    rare_words = list(Vocab.read_bottomn(vocab_dir, lang, word_count).keys())

    # Removing special words that we don't want to be replaced.
    words_to_remove = [sentence_delim.strip('\\') for sentence_delim in SENTENCE_DELIMS.split('|')]
    words_to_remove.extend([PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, BLANK_TOKEN])

    for word_to_remove in words_to_remove:
        if word_to_remove in rare_words:
            rare_words.remove(word_to_remove)

    return rare_words

class TDAugmentor(Augmentor):
    """Class to augment parallel corpora by translation data augmentation
    (parallel) technique (refer: :cite:t:`fadaee2017data`).
    """

    def __init__(self, src_input_path, tgt_input_path, aligner, rare_word_count, src_vocab_dir, augment=True):
        if line_count(src_input_path) != line_count(tgt_input_path):
            raise RuntimeError(ERRORS['corpus_shape'])
        self.doc_count = line_count(src_input_path)

        self.src_lang = path2lang(src_input_path)
        self.tgt_lang = path2lang(tgt_input_path)

        self.src_lm = lm_load(self.src_lang)        # Language model for source language.

        self.rare_words = extract_rare_words(src_vocab_dir, self.src_lang, rare_word_count)

        self.aligner = aligner

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

        logger.info(textwrap.dedent(f'\
            TDAugmentor\n\
            \tdoc_count={self.doc_count}\n\
            \tsrc_input_path={src_input_path}\n\
            \ttgt_input_path={tgt_input_path}\n\
            \tsrc_lang={self.src_lang}\n\
            \ttgt_lang={self.tgt_lang}\n\
            \trare_word_count={rare_word_count}\n\
            \tsrc_vocab_dir={src_vocab_dir}\n\
            Note:\n\
            * Words are 0-indexed.\n\
            * Context consists of all words before current word, but logs show only the last five words.\n\
            * context [a / b] means a is the current word, b is the prediction of the language model using the context.'
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

        # Placeholder list to hold augmented document, will join all sentences
        # in document before returning.
        augmented_src_doc = list()
        augmented_src_doc.append(src_doc[0])    # First word cannot be replaced as it has no context.
        augmented_tgt_doc = tgt_doc.copy()      # Initializing augmented target document with original, replacing words where necessary.

        for src_idx in range(1, len(src_doc)):
            context = ' '.join(src_doc[:src_idx])
            lm_pred = self.src_lm.predict(context, 1)       # Returns context + pred
            lm_pred = lm_pred.split(' ')[-1].strip('▁')     # Removing context from context + pred to get pred. Note: '▁' is NOT the same as underscore ('_').

            # Printing context, [current_word / prediction].
            logger.info(f'context=\'{" ".join(context.split(" ")[max(0, src_idx - 5):])}\' [\'{src_doc[src_idx]}\' / \'{lm_pred}\']')

            if lm_pred in self.rare_words:
                logger.info(f'\t\'{lm_pred}\' is a rare word.')

                # Finding word corresponding to s_{i} in target sentence.
                alignment = self.aligner.align(' '.join(src_doc), ' '.join(tgt_doc)).alignment
                # Finding corresponding word to s_{i} in target sentence.
                tgt_idxs = [t for t, s in alignment if s == src_idx]

                logger.info(f'\talignment(\'{src_doc[src_idx]}\')={[(tgt_idx, tgt_doc[tgt_idx]) for tgt_idx in tgt_idxs]}.')

                if len(tgt_idxs) > 0:
                    # Replacing all words t_{j} aligned to s_{i} with
                    # translation of lm_pred.
                    for tgt_idx in tgt_idxs:
                        alignment = self.aligner.src2tgt(lm_pred)
                        if not alignment is None:
                            augmented_tgt_doc[tgt_idx] = alignment
                            logger.info(f'\t\tReplaced \'{tgt_doc[tgt_idx]}\' with \'{self.aligner.src2tgt(lm_pred)}\', which is the alignment of \'{lm_pred}\'.')
                        else:
                            # Word not aligned.
                            augmented_tgt_doc[tgt_idx] = tgt_doc[tgt_idx]
                            logger.info(f'\t\tDid NOT replace \'{tgt_doc[tgt_idx]}\' as it is NOT aligned to any source word.')

                    augmented_src_doc.append(lm_pred)

                else:
                    # s_{i} is not aligned to any t_{j}, not modifying.
                    augmented_src_doc.append(src_doc[src_idx])

            else:
                logger.info(f'\tDid NOT replace word \'{src_doc[src_idx]}\' at index {src_idx} as \'{lm_pred}\' is NOT a rare word.')
                augmented_src_doc.append(src_doc[src_idx])

        augmented_src_doc = ' '.join(augmented_src_doc)
        augmented_tgt_doc = ' '.join(augmented_tgt_doc)

        logger.info(f'\naugmented_src_doc=\'{augmented_src_doc}\'')
        logger.info(f'augmented_tgt_doc=\'{augmented_tgt_doc}\'')
        logger.info('-' * NUM_LOGGER_DASHES)

        return augmented_src_doc, augmented_tgt_doc

    def __iter__(self):
        pass

    def __len__(self):
        pass