import os

def path2lang(path):
    """Returns language code from extension of path.

    :param path: File whose language code is to be extracted. Note that the file must have extension as language code, for example, `train.en` for English. Refer LANGS below for language codes.
    :type path: str

    :return: Language code.
    :rtype: str
    """

    return os.path.splitext(path)[-1].strip('.')

# Supported languages.
LANGS = [
    'en',   # English
    'hi',   # Hindi
    'mr',   # Marathi
    'ta',   # Tamil
    'te',   # Telugu
    'ur'    # Urdu
]

# Functions that can be used with Preprocess.preprocess.
PREPROC_FUNCS = [
    'pretokenize',
    'normalize'
]

# Make sure raw input corpora are rid of these characters.
INVALID_CHARS = [
    '\t',   # Tabs clash with sep argument used in pandas.read_csv.
    '"'     # Clashes with Python double quotes.
]

ERRORS = {
    # Invalid language code.
    'lang': f'\'lang\' must be one of the language codes in {*LANGS,}. Ensure file extension of corpus files is a language code (for example, \'train.en\' is a valid filename for an English corpus).',

    # Invalid preprocessing function.
    'func': f'funcs must be a str or list of values in {*PREPROC_FUNCS,} or \'all\'.',

    # Mismatch in number of source and target sentences in a preprocessing batch.
    'batch_shape': f'Shape of source and target batches do not match. Check that raw input corpora do not contain any characters among {*INVALID_CHARS,} nor any empty lines.'
}