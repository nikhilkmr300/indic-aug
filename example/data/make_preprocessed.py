import configparser
import sys
import shutil
import os

sys.path.append(os.path.join('..', '..', 'src'))

from preprocess import Preprocess

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(os.path.join('..', '.config'))
    SRC = config['lang']['SRC']
    TGT = config['lang']['TGT']
    BATCH_SIZE = int(config['preproc']['BATCH_SIZE'])   # Number of documents to preprocess at a time for training set.

    if os.path.isdir('preprocessed'):
        shutil.rmtree('preprocessed')
    os.makedirs('preprocessed')

    # Preprocessing raw dataset.
    for dataset in ['train', 'dev', 'test']:
        print(f'Preprocessing {dataset}...')

        src_input_path = os.path.join('raw', f'{dataset}.{SRC}')
        tgt_input_path = os.path.join('raw', f'{dataset}.{TGT}')
        src_output_path = os.path.join('preprocessed', f'{dataset}.{SRC}')
        tgt_output_path = os.path.join('preprocessed', f'{dataset}.{TGT}')

        preproc = Preprocess(src_input_path, tgt_input_path)
        preproc.preprocess(src_output_path, tgt_output_path, BATCH_SIZE, funcs='all')