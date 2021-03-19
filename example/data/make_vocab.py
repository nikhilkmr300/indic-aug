import configparser
import shutil
import os
import sys

import sentencepiece as spm

sys.path.append(os.path.join('..', '..'))
from indic_aug.vocab import *

if __name__ == '__main__':
    config_dir = os.path.join('..')                     # Relative path to directory containing .config.
    config = configparser.ConfigParser()
    config.read(os.path.join(config_dir, '.config'))

    SRC = config['lang']['SRC']
    TGT = config['lang']['TGT']

    PREPROC_DIR = os.path.join(config_dir, config['dir']['PREPROC_DIR'])
    VOCAB_DIR = os.path.join(config_dir, config['dir']['VOCAB_DIR'])

    src_input_path = os.path.join(PREPROC_DIR, f'train.{SRC}')
    tgt_input_path = os.path.join(PREPROC_DIR, f'train.{TGT}')

    SRC_VOCAB_SIZE = int(config['vocab']['SRC_VOCAB_SIZE'])
    TGT_VOCAB_SIZE = int(config['vocab']['TGT_VOCAB_SIZE'])

    if os.path.isdir(VOCAB_DIR):
        shutil.rmtree(VOCAB_DIR)
    os.makedirs(VOCAB_DIR)

    build_vocab(src_input_path, tgt_input_path, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, VOCAB_DIR)