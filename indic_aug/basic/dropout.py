import textwrap

import numpy as np

from ..globals import Augmentor, ERRORS
from ..log import logger, NUM_LOGGER_DASHES
from ..utils import cyclic_read, path2lang, line_count, doc2words

def dropout_aug(doc, p, lang):
    """Performs augmentation on a document by dropout (refer: :cite:t:`iyyer2015deep`).

    :param doc: Document to be augmented.
    :type doc: str
    :param p: Probability of a word to be dropped.
    :type p: float
    :param lang: ISO 639-1 language code of ``doc``.
    :type lang: str

    :return: Augmented document.
    :rtype: str
    """

    augmented_doc = list()

    for idx, word in enumerate(doc2words(doc, lang)):
        if np.random.binomial(1, p):
            logger.info(f'\tDropped word \'{word}\' at index {idx}.')
            # Dropping word with probability p.
            continue
        else:
            augmented_doc.append(word)

    return ' '.join(augmented_doc)

class DropoutAugmentor(Augmentor):
    """Class to augment parallel corpora by dropout (refer:
    :cite:t:`iyyer2015deep`).
    """

    def __init__(self, src_input_path, tgt_input_path, p, augment=True, random_state=1):
        """Constructor method.

        :param src_input_path: Path to aligned source corpus.
        :type src_input_path: str
        :param tgt_input_path: Path to aligned target corpus, corresponding to
            the above source corpus.
        :type tgt_input_path: str
        :param p: Same as for ``dropout_aug``.
        :type p: float
        :param augment: Performs augmentation if ``True``, else returns original
            pair of sentences.
        :type augment: bool
        :param random_state: Seed for the random number generator.
        :type random_state: int
        """

        if line_count(src_input_path) != line_count(tgt_input_path):
            raise RuntimeError(ERRORS['corpus_shape'])
        self.doc_count = line_count(src_input_path)

        np.random.seed(random_state)

        self.src_lang = path2lang(src_input_path)
        self.tgt_lang = path2lang(tgt_input_path)

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
            DropoutAugmentor\n\
            \tdoc_count={self.doc_count}\n\
            \tsrc_input_path={src_input_path}\n\
            \ttgt_input_path={tgt_input_path}\n\
            \tsrc_lang={self.src_lang}\n\
            \ttgt_lang={self.tgt_lang}\n\
            \tp={self.p}\n\
            \trandom_state={random_state}\n\
            Note: Words are 0-indexed.'
        ))
        logger.info('-' * NUM_LOGGER_DASHES)

    def __next__(self):
        # Returning original sentences as they are if self.augment is False.
        if not self.augment:
            return next(self.src_input_file).rstrip('\n'), next(self.tgt_input_file).rstrip('\n')

        src_doc = next(self.src_input_file)
        logger.info(f'src_doc: \'{src_doc}\'')

        augmented_src_doc = dropout_aug(src_doc, self.p, self.src_lang)

        tgt_doc = next(self.tgt_input_file)
        logger.info(f'tgt_doc: \'{tgt_doc}\'')

        augmented_tgt_doc = dropout_aug(tgt_doc, self.p, self.tgt_lang)

        logger.info(f'augmented_src_doc: \'{augmented_src_doc}\'')
        logger.info(f'augmented_tgt_doc: \'{augmented_tgt_doc}\'')
        logger.info('-' * NUM_LOGGER_DASHES)

        return augmented_src_doc, augmented_tgt_doc

    def __iter__(self):
        return self

    def __len__(self):
        return self.doc_count