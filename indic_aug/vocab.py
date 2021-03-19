import configparser
import shutil
import os
import sys

import numpy as np
import pandas as pd
import sentencepiece as spm

from .globals import ERRORS, LANGS, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
from .utils import path2lang

def build_vocab(src_input_path, tgt_input_path, src_vocab_size, tgt_vocab_size, output_dirpath):
    """Generates vocabulary from preprocessed corpus. Use `preprocess.Preprocess` to preprocess raw corpus. Outputs \*.model and \*.vocab files compatible with `sentencepiece`.

    :param src_input_path: Path to preprocessed source corpus.
    :type src_input_path: str
    :param tgt_input_path: Path to preprocessed source corpus.
    :type src_input_path: str
    :param src_vocab_size: Max number of tokens in source vocabulary. Pass -1 to use all tokens.
    :type src_vocab_size: int
    :param tgt_vocab_size: Max number of tokens in target vocabulary. Pass -1 to use all tokens.
    :type tgt_vocab_size: int
    :param output_dirpath: Path to directory where to write \*.model and \*.vocab files.
    :type output_dirpath: str
    """

    src_lang = path2lang(src_input_path)
    tgt_lang = path2lang(tgt_input_path)
    loglevel = 0                            # Corresponds to INFO logging level (refer: https://github.com/google/sentencepiece/blob/master/src/common.h).
    if (not src_lang in LANGS) or (not tgt_lang in LANGS):
        raise ValueError(ERRORS['lang'])

    if src_vocab_size == -1:
        # Using all words as vocabulary.
        spm.SentencePieceTrainer.train(f'--input={src_input_path} --model_prefix={src_lang} --model_type=word --use_all_vocab=true --normalization_rule_name=nmt_nfkc --bos_piece={SOS_TOKEN} --eos_piece={EOS_TOKEN} --unk_piece={UNK_TOKEN} --minloglevel={loglevel}')
    else:
        spm.SentencePieceTrainer.train(f'--input={src_input_path} --model_prefix={src_lang} --model_type=word --vocab_size={src_vocab_size} --normalization_rule_name=nmt_nfkc --bos_piece={SOS_TOKEN} --eos_piece={EOS_TOKEN} --unk_piece={UNK_TOKEN} --minloglevel={loglevel}')

    if tgt_vocab_size == -1:
        # Using all words as vocabulary.
        spm.SentencePieceTrainer.train(f'--input={tgt_input_path} --model_prefix={tgt_lang} --model_type=word --use_all_vocab=true --normalization_rule_name=nmt_nfkc --bos_piece={SOS_TOKEN} --eos_piece={EOS_TOKEN} --unk_piece={UNK_TOKEN} --minloglevel={loglevel}')
    else:
        spm.SentencePieceTrainer.train(f'--input={tgt_input_path} --model_prefix={tgt_lang} --model_type=word --vocab_size={tgt_vocab_size} --normalization_rule_name=nmt_nfkc --bos_piece={SOS_TOKEN} --eos_piece={EOS_TOKEN} --unk_piece={UNK_TOKEN} --minloglevel={loglevel}')

    shutil.move(f'{src_lang}.model', os.path.join(output_dirpath, f'{src_lang}.model'))
    shutil.move(f'{src_lang}.vocab', os.path.join(output_dirpath, f'{src_lang}.vocab'))
    shutil.move(f'{tgt_lang}.model', os.path.join(output_dirpath, f'{tgt_lang}.model'))
    shutil.move(f'{tgt_lang}.vocab', os.path.join(output_dirpath, f'{tgt_lang}.vocab'))

def read_vocab(vocab_path):
    """Reads tokens in vocabulary into a list.

    :param vocab_path: Path to the \*.vocab file to read.
    :type vocab_path: str
    """

    vocab = pd.read_csv(vocab_path, sep='\t', header=None)
    vocab = vocab[0].str.strip('▁').squeeze().tolist()      # Note: '▁' is NOT the same as underscore ('_').

    return vocab

def score2freq(model, words):
    """Converts negative log likelihood score returned by `sentencepiece` to frequency of occurrence of tokens in corpus.

    :param model: `sentencepiece.SentencePieceProcessor` object on which `load` method has been called with a \*.model file.
    :type model: `sentencepiece.SentencePieceProcessor`
    :param words: List of words whose frequencies are to be returned.
    :type words: list

    :return: Dictionary of word to frequency pairs.
    :rtype: dict
    """

    word_ids = [model.encode(word)[0] for word in words]

    freq_dict = dict()
    for word_id in word_ids:
        word = model.IdToPiece(word_id).strip('▁')
        score = model.GetScore(word_id)
        if not score:
            # sentencepiece maps unseen tokens (<unk>, <s>, </s>) to 0 ==> freq = 1, when their freq should be 0.
            freq_dict[word] = 0
        else:
            # sentencepiece outputs negative log likelihood as score. Taking exponent to convert it to frequency.
            freq_dict[word] = np.exp(score)

    return freq_dict

def score2freq_vocab(model_path, vocab_path):
    """Returns frequencies of all words in the vocabulary, given a \*.vocab file.

    :param model_path: Path to \*.model file compatible with `sentencepiece`.
    :type model_path: str
    :param vocab_path: Path to \*.vocab file compatible with `sentencepiece`, corresponding to model at `model_path`.
    :type vocab_path: str

    :return: Dictionary of word to frequency pairs.
    :rtype: dict
    """

    if model_path.split('.')[0] != vocab_path.split('.')[0]:
        raise ValueError('model_path and vocab_path must correspond to the same language.')

    model = spm.SentencePieceProcessor()
    model.load(model_path)

    vocab = read_vocab(vocab_path)

    freq_dict = score2freq(model, vocab)
    # If you are using all the words as vocabulary, this should sum to 1.
    assert sum(freq_dict.values()) <= 1

    return freq_dict