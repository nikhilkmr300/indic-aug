import logging
import warnings

import numpy as np
import pandas as pd
import dill as pickle
from nltk.translate import AlignedSent, Alignment
import matplotlib.pyplot as plt
import seaborn as sns

from .globals import ERRORS, UNK_TOKEN
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

        :param model_type: Variant of IBM Model to use, one of 
            ('ibm1', 'ibm2', 'ibm3'). 
        :type model_type: str
        :param iters: Number of iterations to train the model.
        :type iters: int
        :param max_tokens: Documents with number of tokens greater than
            ``max_tokens`` will not be used for training, pass ``None`` to use
            all documents, defaults to ``None``.
        :type max_tokens: int, optional
        """

        if not model_type in ALIGNER_MODELS:
            raise ValueError(f'model_type must be one of the values in {*ALIGNER_MODELS,}.')

        self.model_type = model_type
        self.iters = iters
        self.max_tokens = max_tokens

    def _load_bitext(self):
        """Loads documents in source and target corpora as
        ``nltk.translate.AlignedSent`` objects.
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
                bitext.append(AlignedSent(tgt_words, src_words))
            elif len(src_words) > self.max_tokens or len(tgt_words) > self.max_tokens:
                logging.info(f'Dropping parallel documents with {len(src_words)} source tokens and {len(tgt_words)} target tokens.')
            else:
                bitext.append(AlignedSent(tgt_words, src_words))

        return bitext

    def train(self, src_input_path, tgt_input_path):
        """Runs training iterations of IBM Model.

        :param src_input_path: Path to source parallel corpus.
        :type src_input_path: str
        :param tgt_input_path: Path to target parallel corpus corresponding to
            above source corpus.
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

    def tgt2src(self, tgt_word):
        """Returns the word in source corpus which has the highest alignment
        score corresponding to ``tgt_word`` in target corpus.

        :param tgt_word: Word in target corpus whose corresponding aligned word
            in source corpus is to be found.
        :type word: str

        :return: Best aligned word in source corpus.
        :rtype: str
        """

        aligned = None
        max_score = 0

        try:
            scores = self.model.translation_table[tgt_word]
        except AttributeError:
            raise AttributeError(ERRORS['call_train'])

        for aligned_candidate, score in scores.items():
            if score > max_score:
                max_score = score
                aligned = aligned_candidate

        return aligned

    def src2tgt(self, src_word):
        """Returns a word in target corpus which has the highest alignment
        score corresponding to ``src_word`` in source corpus.

        :param src_word: Word in source corpus whose corresponding aligned word
            in target corpus is to be found.
        :type word: str

        :return: Best aligned word in target corpus.
        :rtype: str
        """

        aligned = None
        max_score = 0

        for tgt_word, src_words in self.model.translation_table.items():
            if src_word in src_words.keys():
                score = self.model.translation_table[tgt_word][src_word]
                if score > max_score:
                    max_score = score
                    aligned = tgt_word
            else:
                continue

        return aligned

    def align(self, src_sent, tgt_sent):
        """Given a sentence in the source language and a sentence in the target
        language, outputs the values of the alignment function from words in the
        target sentence to words in the source sentence.

        :param src_sent: Source sentence
        :type src_sent: str
        :param tgt_sent: Target sentence
        :type tgt_sent: str

        :return: Aligned sentences
        :rtype: ``nltk.translate.AlignedSent``
        """

        sentence_pair = AlignedSent(tgt_sent.split(' '), src_sent.split(' '))
        self.model.align(sentence_pair)

        return sentence_pair

    def plot_alignment(self, src_sent, tgt_sent, font_family, figsize=(7, 5)):
        """Given a sentence in the source language and a sentence in the target
        language, plots the values of the alignment function from words in the
        target sentence to words in the source sentence as a heatmap.

        :param src_sent: Source sentence
        :type src_sent: str
        :param tgt_sent: Target sentence
        :type tgt_sent: str
        :param font_family: Font family to be passed to ``seaborn.set_theme``.
            Recommended are any of the relevant Sangam MN fonts for MacOS.
        :type font_family: str
        :param figsize: Size of generated heatmap.
        :type figsize: 2-tuple

        :return: Aligned sentences
        :rtype: ``nltk.translate.AlignedSent``
        """

        try:
            self.model.translation_table
        except AttributeError:
            raise AttributeError(ERRORS['call_train'])

        src_words = [None] + src_sent.split(' ')    # Prepending None for target words not aligned to any word.
        tgt_words = tgt_sent.split(' ')

        sns.set_theme(font=font_family)

        alignment_matrix = pd.DataFrame(np.zeros((len(tgt_words), len(src_words))), index=tgt_words, columns=src_words)
        for tgt_word in tgt_words:
            for src_word in src_words:
                alignment_matrix.loc[tgt_word, src_word] = self.model.translation_table[tgt_word][src_word]

        # Changing index None to 'None' so it is visible in the heatmap.
        alignment_matrix.columns.values[0] = 'None'

        plt.figure(figsize=figsize)
        sns.heatmap(alignment_matrix, vmin=0.0, vmax=1.0)

    def serialize(self, path):
        """Saves this object to disk. Use ``dill`` to save serialized object,
        not ``pickle``.

        :param path: Path where to save object.
        :type path: str
        """

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """Loads object from disk. Use ``dill`` to save serialized object, not
        ``pickle``.

        :param path: Path from where to load object.
        :type path: str

        :return: Aligner object stored at path.
        :rtype: ``align.Aligner``
        """

        with open(path, 'rb') as f:
            return pickle.load(f)