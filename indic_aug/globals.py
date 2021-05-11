"""
Module to contain variables that are required in multiple places.

.. data:: LANGS

    List of supported languages.

.. data:: INVALID_CHARS

    List of characters that need to be removed from the raw corpus before using 
    any augmentation. Remove all tabs and double quotes as they clash with the 
    implementation.

.. data:: SENTENCE_DELIMS

    String of pipe (|) separated characters on which to split a document into 
    sentences.

.. data:: PAD_TOKEN

    Token to use for padding sentences.

.. data:: PAD_ID

    Index corresponding to ``PAD_TOKEN``.

.. data:: UNK_TOKEN

    Token to use for unknown words.

.. data:: UNK_ID

    Index corresponding to ``UNK_TOKEN``.

.. data:: SOS_TOKEN

    Token to use for start of sentence.

.. data:: SOS_ID

    Index corresponding to ``SOS_TOKEN``.

.. data:: EOS_TOKEN

    Token to use for end of sentence.

.. data:: EOS_ID

    Index corresponding to ``EOS_TOKEN``.

.. data:: BLANK_TOKEN

    Token to use for blanking words for some augmentation algorithms.

.. data:: BLANK_ID

    Index corresponding to ``BLANK_TOKEN``.
"""

import logging
import os

from abc import ABC, abstractmethod

class Augmentor(ABC):
    """Abstract base class for all augmentor classes."""

    @abstractmethod
    def __next__(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

# Supported languages.
LANGS = [
    'en',   # English
    'hi',   # Hindi
    'mr',   # Marathi
    'ta',   # Tamil
    'te',   # Telugu
]

# Make sure raw input corpora are rid of these characters.
INVALID_CHARS = [
    '\t',                               # Tabs clash with sep argument used in pandas.read_csv.
    "'",                                # Clashes with Python single quotes.
    '"'                                 # Clashes with Python double quotes.
]

SENTENCE_DELIMS = '\.|\?|!|\u0964'      # Characters on which to split sentences in a document.

# Special tokens and their IDs.
PAD_TOKEN = '<pad>'; PAD_ID = 0
UNK_TOKEN = '<unk>'; UNK_ID = 1
SOS_TOKEN = '<s>'; SOS_ID = 2
EOS_TOKEN = '</s>'; EOS_ID = 3
BLANK_TOKEN = '<blank>';                # sentencepiece will assign the next available ID to <blank>.

# Valid augmentation modes for each type of augmentation.
VALID_AUG_MODES = {
    'noising': [                        # Valid modes for noising (Xie paper) augmentation.
        'blank',                        # Replaces word with <blank>.
        'replace',                      # Replaces word with another word from the unigram distribution.
        'absolute_discount',            # Adaptively generates replacement probability using absolute discounting.
        'kneser_ney'                    # Uses absolute discounting while restricting replacement words to a smaller set.
    ],
    'depparse': [                       # Valid modes for dependency parsing augmentation.
        'blank',                        # Replaces word with <blank>.
        'dropout',                      # Deletes word.
        'replace_freq'                  # Replaces word with another word with most similar unigram frequency.
    ],
    'embedding': [                      # Valid modes for embedding augmentation.
        'replace'                       # Replaces word with another word with most similar embedding.
    ]
}

ERRORS = {
    # Invalid language code.
    'lang': f'\'lang\' must be one of the language codes in {*LANGS,}. Ensure file extension of corpus files is a language code (for example, \'train.en\' is a valid filename for an English corpus).',

    # Mismatch in number of source and target sentences in corpus.
    'corpus_shape': f'Shape of source and target corpora do not match. Check that raw input corpora do not contain any characters among {*INVALID_CHARS,} nor any empty lines.',

    # Mismatch in number of source and target sentences in a preprocessing batch.
    'batch_shape': f'Shape of source and target batches do not match. Check that raw input corpora do not contain any characters among {*INVALID_CHARS,} nor any empty lines.',

    # Ensure that root is at index 0 for dependency parsing tree.
    'root_at_0': f'<root> must be at index 0. Ensure that you have prepended dummy <root> to the list of words returned by stanza.models.common.doc.Sentence.words to account for stanza using 1-based indexing for words in sentence.',

    # Word not found.
    'word_not_found': 'Word not found in vocabulary.',

    # Frequency dictionary compulsory with mode 'replace_freq' of 'depparse_aug'.
    'freq_dict_compulsory': 'Passing freq_dict is compulsory with mode=\'replace_freq\' of depparse_aug.',

    # Prev set compulsory with mode 'kneser_ney' of 'noising_aug'.
    'prev_set_compulsory': 'Passing prev_sets is compulsory with mode=\'kneser_ney\' of noising_aug.',
    
    # Cannot generate vocabulary without first preprocessing.
    'call_preprocess': 'Attribute \'prevocab_src_path\' of Preprocessor object does not exist yet. Call \'run_pipeline  \' on Preprocessor object first.',

    # Cannot retrieve alignment without first calling train.
    'call_train': 'Attribute \'model\' of Aligner object does not exist yet. Call \'train\' on Aligner object first.'
}