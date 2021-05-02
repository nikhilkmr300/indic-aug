import csv
import logging
import subprocess
import shutil
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import nltk
import stanza
from indicnlp.tokenize.indic_tokenize import trivial_tokenize_indic
from sacremoses import MosesPunctNormalizer, MosesTruecaser
from indicnlp.normalize.indic_normalize import DevanagariNormalizer, TamilNormalizer, TeluguNormalizer
import sentencepiece as spm
from tqdm import tqdm

from .globals import ERRORS, LANGS, INVALID_CHARS
from .globals import UNK_TOKEN
from .vocab import Vocab
from .utils import path2lang, line_count

class Preprocessor:
    """Class to perform the following:
        1. Preprocess input (tokenizing, normalizing) before feeding to
          sentencepiece.
        2. Generate the vocabulary.
        3. Replace appropriate words in corpus by special tokens.
    """

    def __init__(self, raw_src_path, raw_tgt_path):
        """Constructor method.

        :param raw_src_path: Path to raw source portion of parallel corpus.
        :type raw_src_path: str
        :param raw_tgt_path: Path to raw target portion of parallel corpus.
        :type raw_tgt_path: str
        """

        self.raw_src_path = raw_src_path
        self.raw_tgt_path = raw_tgt_path

        self.src_lang = path2lang(self.raw_src_path)
        self.tgt_lang = path2lang(self.raw_tgt_path)

        self.en_pipeline = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos', verbose=False)

    def pre_vocab(self, prevocab_src_path, prevocab_tgt_path, batch_size):
        """Processing before vocabulary is built ("prevocab processing").

        :param prevocab_src_path: Path to output source corpus after prevocab
            processing.
        :type prevocab_src_path: str
        :param prevocab_tgt_path: Path to output target corpus after prevocab
            processing.
        :type prevocab_tgt_path: str
        :param batch_size: Number of documents to bring into memory at once.
        :type batch_size: int
        """

        def truecase(doc):
            """Truecases a document."""

            # Lowercasing as stanza can tend to treat capitalized words as PROPN by
            # default.
            doc = ' '.join([word.lower() for word in doc.split(' ')])

            truecased = list()
            doc = self.en_pipeline(doc)
            for sent in doc.sentences:
                for i, word in enumerate(sent.words):
                    word.text = word.text.lower()
                    if i == 0 or word.upos == 'PROPN':
                        # Capitalizing first word in sentence and proper nouns.
                        word.text = word.text.capitalize()
                    truecased.append(word.text)

            return ' '.join(truecased)

        def row_tokenize(doc, lang):
            """Performs tokenization on a row of a ``pandas.DataFrame``."""

            # Removing special characters.
            for invalid_char in INVALID_CHARS:
                doc = doc.replace(invalid_char, '')
                doc = doc.replace(invalid_char, '')

            if lang == 'en':
                # doc = truecase(doc)   # Truecasing is too slow.
                return ' '.join(nltk.word_tokenize(doc))
            elif lang in set(LANGS) - {'en'}:
                return ' '.join(trivial_tokenize_indic(doc))
            else:
                raise RuntimeError(ERRORS['lang'])

        def row_normalize(doc, lang):
            """Performs normalization on a row of a ``pandas.DataFrame``."""

            if lang == 'en':
                normalizer = MosesPunctNormalizer(lang=lang)
                truecaser = MosesTruecaser()
                return normalizer.normalize(doc)
            elif lang in {'hi', 'mr'}:
                doc = doc.replace('.', '\u0964')          # Replacing fullstop with poorna virama.
                normalizer = DevanagariNormalizer(lang, remove_nuktas=True, nasals_mode='to_anusvaara_strict', do_normalize_chandras=True, do_normalize_vowel_ending=False)
                return normalizer.normalize(doc)
            elif lang == 'ta':
                normalizer = TamilNormalizer(lang, remove_nuktas=True)
                return normalizer.normalize(doc)
            elif lang == 'te':
                normalizer = TeluguNormalizer(lang, remove_nuktas=True)
                return normalizer.normalize(doc)
            else:
                raise RuntimeError(ERRORS['lang'])

        def batch_process(src_batch, tgt_batch, batch_num):
            """Preprocesses corpus file in batches, useful for corpora with
            large number of documents (no need to bring everything to memory at
            once).

            :param src_batch: Batch of source sentences.
            :type src_batch: ``pandas.Series``
            :param tgt_batch: Batch of target sentences.
            :type tgt_batch: ``pandas.Series``
            :param funcs: Preprocessing functions to apply, refer globals.py for
                allowed functions.
            :type funcs: list
            :param batch_num: Current batch being processed, 1-indexed.
            :type batch_num: int
            """

            # Normalizing.
            src_batch = src_batch.apply(row_normalize, args=(self.src_lang,))
            tgt_batch = tgt_batch.apply(row_normalize, args=(self.tgt_lang,))

            # Tokenizing.
            src_batch = src_batch.apply(row_tokenize, args=(self.src_lang,))
            tgt_batch = tgt_batch.apply(row_tokenize, args=(self.tgt_lang,))

            # Writing to temporary file corresponding to this batch, will be
            # cleaned up by concat_batches.
            tmpfile_path = self.prevocab_src_path + f'.part.{batch_num}'
            if os.path.isfile(tmpfile_path):
                os.remove(tmpfile_path)
            src_batch.to_csv(tmpfile_path, index=False, header=False, quoting=csv.QUOTE_NONE, sep='\t')
            tmpfile_path = self.prevocab_tgt_path + f'.part.{batch_num}'
            if os.path.isfile(tmpfile_path):
                os.remove(tmpfile_path)
            tgt_batch.to_csv(tmpfile_path, index=False, header=False, quoting=csv.QUOTE_NONE, sep='\t')

            # Concatenating generated CSV and accummulated CSV of previous
            # batches.
            concat_batches(batch_num)

        def concat_batches(batch_num):
            """Concatenates current preprocessed batch to CSV containing output
            of previous batches.

            :param batch_num: Batch currently being processed.
            :type batch_num: int
            """

            # Temporary files generated by batch_process, need to be deleted.
            src_tmpfile = f'{self.prevocab_src_path}.part.{batch_num}'
            tgt_tmpfile = f'{self.prevocab_tgt_path}.part.{batch_num}'

            # Shell commands to concatenate temporary files and then delete them.
            subprocess.call(f'cat {src_tmpfile} >> {self.prevocab_src_path}; rm {src_tmpfile}', shell=True)
            subprocess.call(f'cat {tgt_tmpfile} >> {self.prevocab_tgt_path}; rm {tgt_tmpfile}', shell=True)

        # Path to output of prevocab processing.
        self.prevocab_src_path = prevocab_src_path
        self.prevocab_tgt_path = prevocab_tgt_path

        # Finding number of documents to pass as argument to tqdm.
        doc_count = line_count(self.raw_src_path)                       # Number of documents.
        batch_count = int(np.ceil(doc_count / batch_size))              # Theoretical number of batches.

        # Deleting (if exists) and recreating output files.
        subprocess.call(f'rm -f {self.prevocab_src_path}; touch {self.prevocab_src_path}', shell=True)
        subprocess.call(f'rm -f {self.prevocab_tgt_path}; touch {self.prevocab_tgt_path}', shell=True)

        # Applying preprocessing functions in batches.
        self.batch_count = 0
        logger.info(f'Tokenizing... [batch_size={batch_size}, batch_count={batch_count}]')
        for src_batch, tgt_batch in tqdm(zip(
            pd.read_csv(self.raw_src_path, header=None, chunksize=batch_size, squeeze=True, encoding='utf-8', sep='\t'),
            pd.read_csv(self.raw_tgt_path, header=None, chunksize=batch_size, squeeze=True, encoding='utf-8', sep='\t')
        ), total=batch_count):
            if src_batch.shape != tgt_batch.shape:
                raise AssertionError(ERRORS['batch_shape'])

            batch_process(src_batch, tgt_batch, self.batch_count + 1)
            self.batch_count += 1

        # Checking that the number of batches generated by Pandas matches the
        # number of batches according to the formula.
        if self.batch_count != batch_count:
            raise RuntimeError('Something\'s wrong with this implementation.')

    def build_vocab(self, src_vocab_size, tgt_vocab_size, vocab_dir):
        """Generates vocabulary for source and target parallel corpus.

        :param src_vocab_size: Number of words in source vocabulary.
        :type src_vocab_size: int
        :param tgt_vocab_size: Number of words in target vocabulary.
        :type tgt_vocab_size: int
        :param vocab_dir: Path to where ``sentencepiece`` \*.model and \*.vocab
            files to be saved (described in detail in ``vocab.py``).
        :type vocab_dir: str
        """

        try:
            self.prevocab_src_path
        except AttributeError:
            raise AttributeError(ERRORS['call_preprocess'])

        self.vocab_dir = vocab_dir

        if not os.path.exists(self.prevocab_src_path) or not os.path.exists(self.prevocab_tgt_path):
            raise RuntimeError(f'Did you delete the preprocessed files {self.prevocab_src_path} and {self.prevocab_tgt_path}? Run method preprocess again to generate the preprocessed files before calling build_vocab.')

        logger.info('Generating vocabulary...')
        Vocab.build(self.prevocab_src_path, self.prevocab_tgt_path, src_vocab_size, tgt_vocab_size, self.vocab_dir)

    def post_vocab(self, out_src_path, out_tgt_path):
        def postprocess(line, vocab):
            line = [word.strip('\n').strip() for word in line.split()]
            processed_line = list()

            for word in line:
                if word in vocab:
                    processed_line.append(word)
                else:
                    processed_line.append(UNK_TOKEN)

            return ' '.join(processed_line)

        # Source corpus
        try:
            prevocab_src_file = open(self.prevocab_src_path, 'r')
            src_vocab = Vocab.load_vocab(self.vocab_dir, self.src_lang)
        except AssertionError:
            raise AttributeError(ERRORS['call_preprocess'])

        postvocab_src_file = open(out_src_path, 'w')

        logger.info(f'Replacing OOV words in source corpus with {UNK_TOKEN} token...')
        for line in tqdm(prevocab_src_file, total=line_count(self.prevocab_src_path)):
            postvocab_src_file.write(postprocess(line, src_vocab) + '\n')

        # Target corpus
        try:
            prevocab_tgt_file = open(self.prevocab_tgt_path, 'r')
            tgt_vocab = Vocab.load_vocab(self.vocab_dir, self.tgt_lang)
        except AssertionError:
            raise AttributeError(ERRORS['call_preprocess'])

        postvocab_tgt_file = open(out_tgt_path, 'w')

        logger.info(f'Replacing OOV words in target corpus with {UNK_TOKEN} token...')
        for line in tqdm(prevocab_tgt_file, total=line_count(self.prevocab_tgt_path)):
            postvocab_tgt_file.write(postprocess(line, tgt_vocab) + '\n')