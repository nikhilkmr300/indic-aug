import sys
import os

from polyglot.downloader import downloader
from polyglot.mapping import Embedding

from ..log import logger

def fetch_embeddings(lang, polyglot_dir):
    embeddings_filepath = os.path.join(
        polyglot_dir,
        'embeddings2',
        lang,
        'embeddings_pkl.tar.bz2'
    )

    if not os.path.isfile(embeddings_filepath):
        logger.info(f'Could not find embeddings at polyglot_dir=\'{polyglot_dir}\'. Downloading embeddings to {embeddings_filepath}...')
        downloader.download(f'embeddings2.{lang}', download_dir=polyglot_dir, quiet=True)

    return Embedding.load(embeddings_filepath)