import configparser
import shutil
import os
import sys

import sentencepiece as spm

sys.path.append(os.path.join('..', '..', 'src'))

from vocab import *

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(os.path.join('..', '.config'))
    SRC = config['lang']['SRC']
    TGT = config['lang']['TGT']
    SRC_VOCAB_SIZE = config['vocab']['SRC_VOCAB_SIZE']
    TGT_VOCAB_SIZE = config['vocab']['TGT_VOCAB_SIZE']

    if os.path.isdir('vocab'):
        shutil.rmtree('vocab')
    os.makedirs('vocab')

    build_vocab(SRC, TGT, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, 'vocab')