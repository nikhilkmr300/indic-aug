import os
import subprocess

import pandas as pd
import nltk
from indicnlp.tokenize.sentence_tokenize import sentence_split
from indicnlp.tokenize.indic_tokenize import trivial_tokenize

from .globals import ERRORS, UNK_TOKEN, LANGS

def path2lang(path):
    """Returns language code from extension of path.

    :param path: File whose language code is to be extracted. Note that the file
        must have extension as language code, for example, train.en for English.
        Refer globals.py for language codes.
    :type path: str

    :return: Language code.
    :rtype: str
    """

    lang = os.path.splitext(path)[-1].strip('.')

    if not lang in LANGS:
        raise ValueError(ERRORS['lang'])

    return lang

def stanza2list(stanza_sent):
    """Converts ``stanza.models.common.doc.Sentence`` to a list of str tokens,
    by stripping away all the extra stuff.

    :param stanza_sent: Stanza sentence to be converted.
    :type stanza_sent: ``stanza.models.common.doc.Sentence``
    """

    str_sent = list()
    for word in stanza_sent.words:
        str_sent.append(word.text)

    return str_sent

def cyclic_read(filepath):
    """Returns a generator which can read the same file line by line (lazily) arbitrary number of times.

    Using ``open`` to read a file will raise ``StopIteration`` once EOF is
    reached. ``cyclic_read`` will instead loop back to the start of file and
    continue reading indefinitely. Note that it also strips newline characters
    (both '\\n' and '\\r') before returning the line.

    :param filepath: Path to input file.
    :type filepath: str

    :usage: Say you have a file ``sample.txt`` which contains the text 'Line 1',
        'Line 2' and 'Line 3' on three successive lines

    .. code-block: python

    >>> for line in cyclic_read('sample.txt'):
    ...     print(line)
    'Line 1'
    'Line 2'
    'Line 3'
    'Line 1'
    'Line 2'
    'Line 3'

    and so on indefinitely.
    """

    while True:
        with open(filepath, 'r') as f:
            for line in f:
                yield line.rstrip('\n')

def closest_freq(word, freq_dict):
    """Returns the word in ``freq_dict`` that has the closest frequency to that
    of ``word``.

    :param word: Word whose closest frequency word is to be found.
    :type word: str
    :param freq_dict: Word to frequency mapping as returned by
        ``vocab.freq2dict_vocab``.
    :type freq_dict: dict

    :return: Word with closest frequency to that of ``word``.
    :rtype: str
    """

    if not word in freq_dict.keys():
        word = UNK_TOKEN

    # Converting frequency dictionary to dataframe for easier handling.
    freq_df = pd.DataFrame(
        [[word, freq] for word, freq in freq_dict.items()],
        columns=['word', 'freq']
    ).sort_values(by='freq', ascending=False).reset_index(drop=True)

    # Index of desired word
    word_idx = freq_df[freq_df['word'] == word].index

    # Since freq_df is sorted, word of closest frequency will either be previous word or next word.
    if word_idx == 0:
        return freq_df.loc[word_idx + 1, 'word'].values[0]
    elif word_idx == len(freq_df) - 1:
        return freq_df.loc[word_idx - 1, 'word'].values[0]
    else:
        word_minus_freq = freq_df.loc[word_idx - 1, 'freq'].values[0]
        word_freq = freq_df.loc[word_idx, 'freq'].values[0]
        word_plus_freq = freq_df.loc[word_idx + 1, 'freq'].values[0]

        if word_minus_freq - word_freq < word_freq - word_plus_freq:
            # Previous word is closer.
            return freq_df.loc[word_idx - 1, 'word'].values[0]
        else:
            # Next word is closer.
            return freq_df.loc[word_idx + 1, 'word'].values[0]

def line_count(path):
    """Returns the number of lines in a file.

    :param path: Path to file.
    :type path: str

    :return: Number of lines in file.
    :rtype: int
    """

    if not os.path.isfile(path):
        raise FileNotFoundError

    process = subprocess.Popen(['wc', '-l', path], stdout=subprocess.PIPE)

    return int(process.communicate()[0].strip().split()[0])

def doc2sents(doc, lang):
    """Splits a document into sentences. Wrapper around ``nltk.sent_tokenize``
    and ``indicnlp.tokenize.sentence_tokenize.sentence_split``.

    :param doc: Document to be split into sentences.
    :type doc: str
    :param lang: ISO 639-1 language code of ``doc``.
    :type lang: str

    :return: List of sentences in ``doc``.
    :rtype: list
    """

    doc = doc.strip('\n\t ')

    if lang == 'en':
        return nltk.sent_tokenize(doc)
    elif lang in LANGS:
        return sentence_split(doc, lang=lang)
    else:
        raise ValueError(ERRORS['lang'])

def doc2words(doc, lang):
    """Splits a document into words. Wrapper around ``nltk.word_tokenize`` and
    ``indicnlp.tokenize.indic_tokenize.trivial_tokenize``.

    :param doc: Document to be split into words.
    :type doc: str
    :param lang: ISO 639-1 language code of ``doc``.
    :type lang: str

    :return: List of words in ``doc``.
    :rtype: list
    """

    doc = doc.strip('\n\t ')

    if lang == 'en':
        return nltk.word_tokenize(doc)
    elif lang in LANGS:
        return trivial_tokenize(doc, lang=lang)
    else:
        raise ValueError(ERRORS['lang'])

def sent2words(doc, lang):
    """Splits a sentence into words. Wrapper around ``nltk.word_tokenize`` and
    ``indicnlp.tokenize.indic_tokenize.trivial_tokenize``.

    Same as ``doc2words``, however have kept separate for readability reasons
    (document vs sentence).

    :param doc: Document to be split into words.
    :type doc: str
    :param lang: ISO 639-1 language code of ``sent``.
    :type lang: str

    :return: List of words in ``sent``.
    :rtype: list
    """

    doc = doc.strip('\n\t ')

    if lang == 'en':
        return nltk.word_tokenize(doc)
    elif lang in LANGS:
        return trivial_tokenize(doc, lang=lang)
    else:
        raise ValueError(ERRORS['lang'])