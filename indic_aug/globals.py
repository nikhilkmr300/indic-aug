import os
from abc import ABC, abstractmethod

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
    '\t',                           # Tabs clash with sep argument used in pandas.read_csv.
    '"'                             # Clashes with Python double quotes.
]

# Special tokens and their IDs.
PAD_TOKEN = '<pad>'; PAD_ID = 0
UNK_TOKEN = '<unk>'; UNK_ID = 1
SOS_TOKEN = '<s>'; SOS_ID = 2
EOS_TOKEN = '</s>'; EOS_ID = 3
BLANK_TOKEN = '<blank>';            # sentencepiece will assign the next available ID to <blank>.

# Valid augmentation modes for each type of augmentation.
VALID_AUG_MODES = {
    'depparse': [                   # Valid modes for dependency parsing augmentation.
        'blank',                    # Replaces word with <blank>.
        'dropout',                  # Delete word.
        'replace'                   # Replaces word with another word with most similar unigram frequency.
    ],
    'embedding': [                  # Valid modes for embedding augmentation.
        'replace'                   # Replaces word with another word with most similar embedding.
    ]
}

ERRORS = {
    # Invalid language code.
    'lang': f'\'lang\' must be one of the language codes in {*LANGS,}. Ensure file extension of corpus files is a language code (for example, \'train.en\' is a valid filename for an English corpus).',

    # Invalid preprocessing function.
    'func': f'funcs must be a str or list of values in {*PREPROC_FUNCS,} or \'all\'.',

    # Mismatch in number of source and target sentences in a preprocessing batch.
    'batch_shape': f'Shape of source and target batches do not match. Check that raw input corpora do not contain any characters among {*INVALID_CHARS,} nor any empty lines.',

    # Ensure that root is at index 0 for dependency parsing tree.
    'root_at_0': f'<root> must be at index 0. Ensure that you have prepended dummy <root> to the list of words returned by stanza.models.common.doc.Sentence.words to account for stanza using 1-based indexing for words in sentence.',

    # Invalid augmentation mode.
    'invalid_aug_mode': f'Invalid augmentation mode. Refer to globals for allowed values.',

    # Word not found.
    'word_not_found': 'Word not found in vocabulary.'
}