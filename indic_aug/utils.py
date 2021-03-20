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
    """Converts `stanza.models.common.doc.Sentence` to a list of str tokens, by stripping away all the extra stuff.

    :param stanza_sent: Stanza sentence to be converted.
    :type stanza_sent: `stanza.models.common.doc.Sentence`
    """

    str_sent = list()
    for word in stanza_sent.words:
        str_sent.append(word.text)

    return str_sent