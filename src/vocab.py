import shutil
import os

import sentencepiece as spm

from globals import LANGS, ERRORS

def build_vocab(src_lang, tgt_lang, src_vocab_size, tgt_vocab_size, output_dirpath):
    if (not src_lang in LANGS) or (not tgt_lang in LANGS):
        raise ValueError(ERRORS['lang'])

    spm.SentencePieceTrainer.train(f'--input=preprocessed/train.{src_lang} --model_prefix={src_lang} --model_type=word --vocab_size={src_vocab_size}')
    spm.SentencePieceTrainer.train(f'--input=preprocessed/train.{tgt_lang} --model_prefix={tgt_lang} --model_type=word --vocab_size={tgt_vocab_size}')

    shutil.move(f'{src_lang}.model', os.path.join(output_dirpath, f'{src_lang}.model'))
    shutil.move(f'{src_lang}.vocab', os.path.join(output_dirpath, f'{src_lang}.vocab'))
    shutil.move(f'{tgt_lang}.model', os.path.join(output_dirpath, f'{tgt_lang}.model'))
    shutil.move(f'{tgt_lang}.vocab', os.path.join(output_dirpath, f'{tgt_lang}.vocab'))