import csv
import pathlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import nltk
from indicnlp.tokenize.indic_tokenize import trivial_tokenize_indic, trivial_tokenize_urdu
from indicnlp.tokenize.indic_detokenize import trivial_detokenize
from indicnlp.normalize.indic_normalize import DevanagariNormalizer, TamilNormalizer, TeluguNormalizer
from sacremoses.normalize import MosesPunctNormalizer
from tqdm import tqdm

from .globals import path2lang, ERRORS, LANGS, PREPROC_FUNCS

def pretokenize_sent(sent, lang):
    """Returns tokenized form of a sentence.

    :param sent: Sentence to be tokenized.
    :type sent: str
    :param lang: Language code (refer globals.py) of sentence.
    :type lang: str

    :return: Tokenized sentence.
    :rtype: str
    """

    if lang == 'en':
        return ' '.join(nltk.word_tokenize(sent))
    elif lang in {'hi', 'mr', 'ta', 'te'}:
        return ' '.join(trivial_tokenize_indic(sent))
    elif lang == 'ur':
        return ' '.join(trivial_tokenize_urdu(sent))
    else:
        raise RuntimeError(ERRORS['lang'])

def normalize_sent(sent, lang):
    """Returns normalized form of sentence.

    :param sent: Sentence to be tokenized.
    :type sent: str
    :param lang: Language code (refer globals.py) of sentence.
    :type lang: str.

    :return: Normalized sentence.
    :rtype: str
    """

    if lang == 'en':
        normalizer = MosesPunctNormalizer(lang=lang)
        return normalizer.normalize(sent)
    elif lang in {'hi', 'mr'}:
        normalizer = DevanagariNormalizer(lang, remove_nuktas=True, nasals_mode='to_anusvaara_strict', do_normalize_chandras=True, do_normalize_vowel_ending=False)
        return normalizer.normalize(sent)
    elif lang == 'ta':
        normalizer = TamilNormalizer(lang, remove_nuktas=True)
        return normalizer.normalize(sent)
    elif lang == 'te':
        normalizer = TeluguNormalizer(lang, remove_nuktas=True)
        return normalizer.normalize(sent)
    elif lang == 'ur':
        return sent
    else:
        raise RuntimeError(ERRORS['lang'])

class Preprocess:
    """Preprocessing raw input."""

    def __init__(self, src_input_path, tgt_input_path):
        """Constructor method."""

        self.src_input_path = src_input_path                    # Path to input source sentences.
        self.tgt_input_path = tgt_input_path                    # Path to input target sentences.

        self.src_lang = self._get_lang(self.src_input_path)     # Source language
        self.tgt_lang = self._get_lang(self.tgt_input_path)     # Target language

    def _get_lang(self, path):
        """Returns language of text in path. Ensure your files are named in the format <filename>.<lang_code>. Refer globals.py for language codes.

        :param path: Path to corpus file.
        :type path: str

        :return: Language code of file.
        :rtype: str
        """

        lang = path2lang(path)
        if not lang in LANGS:
            raise RuntimeError(ERRORS['lang'])

        return lang

    def _batch_process(self, src_batch, tgt_batch, funcs, batch_num):
        """Preprocesses corpus file in batches, useful for corpora with large number of documents.

        :param src_batch: Batch of source sentences.
        :type src_batch: `pandas.Series`
        :param tgt_batch: Batch of target sentences.
        :type tgt_batch: `pandas.Series`
        :param funcs: Preprocessing functions to apply, refer globals.py for allowed functions.
        :type funcs: list
        :param batch_num: Current batch being processed, 1-indexed.
        :type batch_num: int
        """

        # Tokenizing.
        if 'pretokenize' in funcs:
            src_batch = src_batch.apply(pretokenize_sent, args=(self.src_lang,))
            tgt_batch = tgt_batch.apply(pretokenize_sent, args=(self.tgt_lang,))

        # Normalizing.
        if 'normalize' in funcs:
            src_batch = src_batch.apply(normalize_sent, args=(self.src_lang,))
            tgt_batch = tgt_batch.apply(normalize_sent, args=(self.tgt_lang,))

        # Writing to temporary file corresponding to this batch, will be cleaned up by self._concat.
        if not self.src_output_path is None:
            tmpfile_path = self.src_output_path + f'.part.{batch_num}'
            if os.path.isfile(tmpfile_path):
                os.remove(tmpfile_path)
            src_batch.to_csv(tmpfile_path, index=False, header=False, quoting=csv.QUOTE_NONE, sep='\t')
        if not self.tgt_output_path is None:
            tmpfile_path = self.tgt_output_path + f'.part.{batch_num}'
            if os.path.isfile(tmpfile_path):
                os.remove(tmpfile_path)
            tgt_batch.to_csv(tmpfile_path, index=False, header=False, quoting=csv.QUOTE_NONE, sep='\t')
        self._concat(batch_num)

    def _concat(self, batch_num):
        """Concatenates current preprocessed batch to CSV containing output of previous batches.

        :param batch_num: Batch currently being processed.
        :type batch_num: int
        """

        # Temporary files generated by self._batch_process, need to be deleted.
        src_tmpfile = f'{self.src_output_path}.part.{batch_num}'
        tgt_tmpfile = f'{self.tgt_output_path}.part.{batch_num}'

        # Shell commands to concatenate temporary files and then delete them.
        subprocess.call(f'cat {src_tmpfile} >> {self.src_output_path}; rm {src_tmpfile}', shell=True)
        subprocess.call(f'cat {tgt_tmpfile} >> {self.tgt_output_path}; rm {tgt_tmpfile}', shell=True)

    def preprocess(self, src_output_path, tgt_output_path, batch_size, funcs='all'):
        """Performs preprocessing of input corpus.

        :param src_output_path: Path to preprocessed source corpus.
        :type src_output_path: str
        :param tgt_output_path: Path to preprocessed target corpus.
        :type tgt_output_path: str
        :param batch_size: Number of documents to be brought into memory at a time.
        :type batch_size: int
        :param funcs: Preprocessing functions to apply to raw corpus (refer globals.py for allowed functions).
        :type funcs: str, list
        """

        self.src_output_path = src_output_path      # Path to preprocessed source sentences.
        self.tgt_output_path = tgt_output_path      # Path to preprocessed target sentences.
        self.batch_count = 0                        # Number of batches as generated by pandas.read_csv.

        if funcs == 'all':
            funcs = PREPROC_FUNCS
        elif type(funcs) == str:
            if funcs in PREPROC_FUNCS:
                funcs = [funcs]
            else:
                raise ValueError(ERRORS['func'])
        for func in funcs:
            if not func in PREPROC_FUNCS:
                raise ValueError(ERRORS['func'])

        # Finding number of documents to pass as argument to tqdm.
        process = subprocess.Popen(['wc', '-l', self.src_input_path], stdout=subprocess.PIPE)
        doc_count = int(process.communicate()[0].strip().split()[0])    # Number of documents.
        batch_count = int(np.ceil(doc_count / batch_size))              # Theoretical number of batches.

        # Deleting (if exists) and recreating output files.
        subprocess.call(f'rm -f {src_output_path}; touch {src_output_path}', shell=True)
        subprocess.call(f'rm -f {tgt_output_path}; touch {tgt_output_path}', shell=True)

        # Applying preprocessing functions in batches.
        print(f'[batch_size={batch_size}, batch_count={batch_count}, funcs={*funcs,}]')
        for src_batch, tgt_batch in tqdm(zip(
            pd.read_csv(self.src_input_path, header=None, chunksize=batch_size, squeeze=True, encoding='utf-8', sep='\t'),
            pd.read_csv(self.tgt_input_path, header=None, chunksize=batch_size, squeeze=True, encoding='utf-8', sep='\t')
        ), total=batch_count):
            if src_batch.shape != tgt_batch.shape:
                raise AssertionError(ERRORS['batch_shape'])

            self._batch_process(src_batch, tgt_batch, funcs, self.batch_count + 1)
            self.batch_count += 1

        # Checking that the number of batches generated by Pandas matches the number of batches according to the formula.
        if self.batch_count != batch_count:
            raise RuntimeError('Something\'s wrong with this implementation.')