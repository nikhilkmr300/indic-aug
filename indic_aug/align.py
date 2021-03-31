import logging

from nltk.translate import AlignedSent
import dill as pickle

from .globals import ERRORS
from .utils import path2lang, line_count, doc2words

ALIGNER_MODELS = [
    'ibm1',
    'ibm2',
    'ibm3'
]

class Aligner:
    """Class to perform bitext word alignment using the IBM Models.
    """

    def __init__(self, model_type, iters, max_tokens=None):
        """Constructor method.

        :param model_type: Variant of IBM Model to use, one of ('ibm1', 'ibm2', 'ibm3').
        :type model_type: str
        :param iters: Number of iterations to train the model.
        :type iters: int
        :param max_tokens: Documents with number of tokens greater than ``max_tokens`` will not be used for training, pass None to use all documents, defaults to None.
        :type max_tokens: int, optional
        """

        if not model_type in ALIGNER_MODELS:
            raise ValueError(f'model_type must be one of the values in {*ALIGNER_MODELS,}.')

        self.model_type = model_type
        self.iters = iters
        self.max_tokens = max_tokens

    def _load_bitext(self):
        """Loads documents in source and target corpora as ``nltk.translate.AlignedSent`` objects.
        """

        if line_count(self.src_input_path) != line_count(self.tgt_input_path):
            raise RuntimeError(ERRORS['corpus_shape'])

        bitext = list()

        src_file = open(self.src_input_path, 'r')
        tgt_file = open(self.tgt_input_path, 'r')

        for src_doc, tgt_doc in zip(src_file, tgt_file):
            src_words = doc2words(src_doc, self.src_lang)
            tgt_words = doc2words(tgt_doc, self.tgt_lang)

            if self.max_tokens is None:
                bitext.append((src_words, tgt_words))
            elif len(src_words) > self.max_tokens or len(tgt_words) > self.max_tokens:
                logging.info(f'Dropping parallel documents with {len(src_words)} source tokens and {len(tgt_words)} target tokens.')
            else:
                bitext.append(AlignedSent(src_words, tgt_words))

        return bitext

    def train(self, src_input_path, tgt_input_path):
        """Runs training iterations of IBM Model.

        :param src_input_path: Path to source parallel corpus.
        :type src_input_path: str
        :param tgt_input_path: Path to target parallel corpus corresponding to above source corpus.
        :type tgt_input_path: str
        """

        self.src_input_path = src_input_path
        self.tgt_input_path = tgt_input_path

        self.src_lang = path2lang(self.src_input_path)
        self.tgt_lang = path2lang(self.tgt_input_path)

        bitext = self._load_bitext()

        if self.model_type == 'ibm1':
            from nltk.translate.ibm1 import IBMModel1
            self.model = IBMModel1(bitext, self.iters)

        elif self.model_type == 'ibm2':
            from nltk.translate.ibm2 import IBMModel2
            self.model = IBMModel2(bitext, self.iters)

        elif self.model_type == 'ibm3':
            from nltk.translate.ibm3 import IBMModel3
            self.model = IBMModel3(bitext, self.iters)

    def get_aligned(self, word):
        """Returns the aligned word in target corpus corresponding to ``word`` in source corpus.

        :param word: Word in source corpus whose corresponding aligned word in target corpus is to be found.
        :type word: str

        :return: Aligned word.
        :rtype: str
        """

        aligned = None
        max_score = 0
        scores = self.model.translation_table[word]

        for aligned_candidate, score in scores.items():
            if score > max_score:
                max_score = score
                aligned = aligned_candidate

        return aligned

    def serialize(self, path):
        """Saves this object to disk. Use ``dill`` to load serialized object, not ``pickle``.

        :param path: Path where to save object.
        :type path: str
        """

        with open(path, 'wb') as f:
            pickle.dump(self, f)