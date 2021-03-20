import os

def path2lang(path):
    """Returns language code from extension of path.

    :param path: File whose language code is to be extracted. Note that the file must have extension as language code, for example, train.en for English. Refer globals.py for language codes.
    :type path: str

    :return: Language code.
    :rtype: str
    """

    return os.path.splitext(path)[-1].strip('.')

def stanza2list(stanza_sent):
    """Converts ``stanza.models.common.doc.Sentence`` to a list of str tokens, by stripping away all the extra stuff.

    :param stanza_sent: Stanza sentence to be converted.
    :type stanza_sent: ``stanza.models.common.doc.Sentence``
    """

    str_sent = list()
    for word in stanza_sent.words:
        str_sent.append(word.text)

    return str_sent

def cyclic_read(filepath):
    """Returns a generator which can read the same file line by line (lazily) arbitrary number of times.

    Using ``open`` to read a file will raise ``StopIteration`` once EOF is reached. ``cyclic_read`` will instead loop back to the start of file and continue reading indefinitely. Note that it also strips newline characters (both '\\n' and '\\r') before returning the line.

    :param filepath: Path to input file.
    :type filepath: str

    :usage: Say you have a file ``sample.txt`` which contains the following text

    .. code-block: text

    Line 1
    Line 2
    Line 3

    .. code-block: python

    >>> for line in cyclic_read('sample.txt'):
    ...     print(line)
    'Line 1'
    'Line 2'
    'Line 3'
    'Line 1'
    'Line 2'
    'Line 3'

    and so on indefinitely...

    Alternatively, you could use ``next`` as you would on a generator as follows

    .. code-block: python

    >>> lines = cyclic_read('sample.txt')
    >>> next(lines)
    'Line 1'
    >>> next(lines)
    'Line 2'
    >>> next(lines)
    'Line 3'
    >>> next(lines)
    'Line 1'
    >>> next(lines)
    'Line 2'

    """

    while True:
        with open(filepath, 'r') as f:
            for line in f:
                yield line.rstrip('\n')