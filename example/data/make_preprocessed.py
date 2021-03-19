import configparser
import sys
import shutil
import os

sys.path.append(os.path.join('..', '..'))
from indic_aug.preprocess.preprocess import Preprocess

if __name__ == '__main__':
    config_dir = os.path.join('..')                     # Relative path to directory containing .config.
    config = configparser.ConfigParser()
    config.read(os.path.join('..', '.config'))

    SRC = config['lang']['SRC']
    TGT = config['lang']['TGT']

    BATCH_SIZE = int(config['preproc']['BATCH_SIZE'])   # Number of documents to preprocess at a time for training set.

    RAW_DIR = os.path.join(config_dir, config['dir']['RAW_DIR'])
    PREPROC_DIR = os.path.join(config_dir, config['dir']['PREPROC_DIR'])

    if os.path.isdir(PREPROC_DIR):
        shutil.rmtree(PREPROC_DIR)
    os.makedirs(PREPROC_DIR)

    # Preprocessing raw dataset.
    for dataset in ['train', 'dev', 'test']:
        print(f'Preprocessing {dataset}...')

        src_input_path = os.path.join(RAW_DIR, f'{dataset}.{SRC}')
        tgt_input_path = os.path.join(RAW_DIR, f'{dataset}.{TGT}')
        src_output_path = os.path.join(PREPROC_DIR, f'{dataset}.{SRC}')
        tgt_output_path = os.path.join(PREPROC_DIR, f'{dataset}.{TGT}')

        preproc = Preprocess(src_input_path, tgt_input_path)
        preproc.preprocess(src_output_path, tgt_output_path, BATCH_SIZE, funcs='all')