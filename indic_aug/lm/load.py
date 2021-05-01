import asyncio
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import shutil
import os

from inltk.config import LMConfigs
from inltk.download_assets import download_file
from fastai.text import load_learner

from ..globals import LANGS, ERRORS

def fetch(lang):
    """Fetches pretrained language model (refer: :cite:t:`arora2020inltk`) from
    remote Dropbox source.

    :param lang: ISO 639-1 code for language for which to retrieve language
        model.
    :type lang: str
    """

    if not lang in LANGS:
        raise ValueError(ERRORS['lang'])

    # Directory where to save pretrained language model.
    model_dir = os.path.join(
        Path(__file__).parent,
        lang
    )
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)

    with redirect_stdout(io.StringIO(os.devnull)):
        # Downloading language model and tokenizer.
        config = LMConfigs(lang).get_config()
        asyncio.run(download_file(config['lm_model_url'], Path(model_dir), config['lm_model_file_name']))
        asyncio.run(download_file(config['tokenizer_model_url'], Path(model_dir), config['tokenizer_model_file_name']))

def load(lang):
    """Loads language model if already downloaded, else downloads and loads.

    :param lang: ISO 639-1 code for language for which to load language model.
    :type lang: str

    :return: Language model
    :rtype: ``fastai.text.learner.LanguageLearner``
    """

    model_dir = os.path.join(
        Path(__file__).parent,
        lang
    )

    # Downloading language model if  not already downloaded.
    if not os.path.exists(model_dir):
        fetch(lang)

    with redirect_stderr(io.StringIO(os.devnull)):
        model = load_learner(model_dir)

    return model