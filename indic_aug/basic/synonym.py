import random
import re
import sys

import numpy as np
from nltk.corpus import wordnet as wn
import stanza
import pyiwn

from ..globals import Augmentor, ERRORS, SENTENCE_DELIMS
from ..utils import cyclic_read, path2lang, line_count, doc2words

def convert_pos(stanza_pos, to):
    """Converts Universal Dependencies parts of speech to WordNet/IndoWordNet parts of speech.

    :param stanza_pos: POS defined by Universal Dependencies.
    :type stanza_pos: str
    :param to: POS to convert to, can be 'wn' for WordNet and 'iwn' for IndoWordNet.
    :type to: str

    :return: POS in WordNet/IndoWordNet format.
    :rtype: str if ``to`` is 'wn', ``pyiwn.iwn.PosTag`` if ``to`` is 'iwn'
    """

    # Map from Universal Dependencies POS to WordNet POS.
    ud2wn_map = {
        'ADJ': wn.ADJ,
        'ADP': None,
        'ADV': wn.ADV,
        'AUX': None,
        'CCONJ': None,
        'DET': None,
        'INTJ': None,
        'NOUN': wn.NOUN,
        'NUM': None,
        'PART': None,
        'PRON': None,
        'PROPN': wn.NOUN,
        'PUNCT': None,
        'SCONJ': None,
        'SYM': None,
        'VERB': wn.VERB,
        'X': None
    }
    # Map from Universal Dependencies POS to IndoWordNet POS.
    ud2iwn_map = {
        'ADJ': pyiwn.iwn.PosTag.ADJECTIVE,
        'ADP': None,
        'ADV': pyiwn.iwn.PosTag.ADVERB,
        'AUX': None,
        'CCONJ': None,
        'DET': None,
        'INTJ': None,
        'NOUN': pyiwn.iwn.PosTag.NOUN,
        'NUM': None,
        'PART': None,
        'PRON': None,
        'PROPN': pyiwn.iwn.PosTag.NOUN,
        'PUNCT': None,
        'SCONJ': None,
        'SYM': None,
        'VERB': pyiwn.iwn.PosTag.VERB,
        'X': None
    }

    if not stanza_pos in ud2iwn_map.keys():
        raise ValueError(f'Invalid value of parameter \'stanza_pos\'. Valid values for parameter \'stanza_pos\' to convert_pos are values in {ud2iwn_map.keys()}.')

    if to == 'wn':
        return ud2wn_map[stanza_pos]
    elif to == 'iwn':
        return ud2iwn_map[stanza_pos]
    else:
        raise ValueError(f'Invalid value of parameter \'to\'. Valid values for parameter \'to\' are values in (\'wn\', \'iwn\'. Pass \'wn\' to convert from Universal Dependencies POS to WordNet POS (for English) and \'iwn\' to convert from Universal Dependencies POS to IndoWordNet POS (for Indian languages).')

def get_synonyms(word, pipeline, net):
    """Returns a list of synonyms of ``word``.

    :param word: Word whose synonym is required.
    :type word: str
    :param pipeline: ``stanza.Pipeline`` corresponding to language of word.
    :type pipeline: `stanza.Pipeline`
    :param net: Thesaurus (word net) to use. Use nltk.corpus.wordnet for English and pyiwn.iwn.IndoWordNet() for Indian languages.
    :type net: `nltk.corpus.reader.wordnet.WordNetCorpusReader` for English and `pyiwn.iwn.IndoWordNet` for Indian languages.

    :return: List of synonyms for ``word``.
    :rtype: list
    """

    if len(word.split(' ')) != 1:
        raise ValueError(f'Parameter \'word\' to get_synonyms must be a str with no spaces.')

    lang = pipeline.lang

    # POS of word whose synonyms are required in Universal Dependencies format, used by stanza.
    stanza_pos = pipeline(word).sentences[0].words[0].upos
    # POS of word whose synonyms are required in WordNet or IndoWordNet format, used by nltk/pyiwn.
    converted_pos = convert_pos(stanza_pos, 'wn') if lang == 'en' else convert_pos(stanza_pos, 'iwn')

    if converted_pos is None:
        # If converted_pos is None, unknown POS, nltk/pyiwn cannot handle, return word as its own synonym.
        return [word]

    synonyms = list()

    try:
        # Synonym sets corresponding to `word` and have the same POS as `word`.
        synsets = net.synsets(word, pos=converted_pos)
    except KeyError as e:
        # If word does not exist in IndoWordNet, return word as its own synonym.
        return [word]
    except LookupError as e:
        # Downloading wordnet.
        print(f'Downloading nltk.corpus.wordnet. Rerun the program once download is complete.')
        import nltk
        nltk.download('wordnet')
        sys.exit()

    # Extracting synonyms which have the same POS as input word.
    for synset in synsets:
        synonyms.extend(synset.lemma_names())

    # Removing occurrences of word itself from its list of synonyms.
    synonyms = [synonym for synonym in synonyms if synonym != word]

    if len(synonyms) == 0:
        # If removing occurrences of word itself led to synonyms being empty, return word as its own synonym.
        synonyms = [word]

    return synonyms

def synonym_aug(doc, pipeline, net, p):
    """Performs augmentation on a document by replacing with synonyms (refer: :cite:t:`wei2019eda`).

    :param doc: Document to be augmented.
    :type doc: str
    :param pipeline: Same as for ``get_synonyms``.
    :type pipeline: `stanza.Pipeline`
    :param net: Same as for ``get_synonyms``.
    :type net: `nltk.corpus.reader.wordnet.WordNetCorpusReader` for English and `pyiwn.iwn.IndoWordNet` for Indian languages.
    :param p: Probability of a word to be replaced by one of its synonyms.
    :type p: float

    :return: Augmented document.
    :rtype: str
    """

    augmented_doc = list()

    lang = pipeline.lang

    for word in doc2words(doc, lang):
        if word in set(re.split('|', SENTENCE_DELIMS)):
            # Not replacing punctuations.
            continue

        if np.random.binomial(1, p):
            # Randomly sampling a synonym from list of synonyms of word.
            synonyms = get_synonyms(word, pipeline, net)
            if not len(synonyms):
                # If no synonyms found, replace word with itself.
                sampled_synonym = word
            else:
                # Else replace word with a randomly sampled synonym.
                sampled_synonym = random.sample(synonyms, 1)[0]
            augmented_doc.append(sampled_synonym)
        else:
            augmented_doc.append(word)

    return ' '.join(augmented_doc)

class SynonymAugmentor(Augmentor):
    """Class to augment parallel corpora by synonym augmentation technique (refer: :cite:t:`wei2019eda`)."""

    def __init__(self, src_input_path, tgt_input_path, p, augment=True, random_state=1):
        """Constructor method.

        :param src_input_path: Path to aligned source corpus.
        :type src_input_path: str
        :param tgt_input_path: Path to aligned target corpus, corresponding to the above source corpus.
        :type tgt_input_path: str
        :param p: Same as for ``synonym_aug``.
        :type p: float
        :param augment: Performs augmentation if ``True``, else returns original pair of sentences.
        :type augment: bool
        :param random_state: Seed for the random number generator.
        :type random_state: int
        """

        if line_count(src_input_path) != line_count(tgt_input_path):
            raise RuntimeError(ERRORS['corpus_shape'])
        self.doc_count = line_count(src_input_path)

        random.seed(random_state)                   # synonym_aug uses random from standard library.
        np.random.seed(random_state)

        self.src_lang = path2lang(src_input_path)
        self.tgt_lang = path2lang(tgt_input_path)

        self.augment = augment

        if self.augment:
            # If augment is True, can perform arbitrary number of augmentations by cycling through all the sentences in the corpus repeatedly.
            self.src_input_file = cyclic_read(src_input_path)
            self.tgt_input_file = cyclic_read(tgt_input_path)
        else:
            # Else does one pass through the corpus and stops.
            self.src_input_file = open(src_input_path)
            self.tgt_input_file = open(tgt_input_path)

        self.p = p

        # Setting up stanza pipelines for source and target languages, for use with synonym_aug.
        self.src_pipeline = stanza.Pipeline(lang=self.src_lang)
        self.tgt_pipeline = stanza.Pipeline(lang=self.tgt_lang)

        # Setting up objects to use WordNet/IndoWordNet.
        self.src_net = wn if self.src_lang == 'en' else pyiwn.IndoWordNet()
        self.tgt_net = wn if self.tgt_lang == 'en' else pyiwn.IndoWordNet()

    def __next__(self):
        """Returns a pair of sentences on every call using a generator. Does a lazy load of the data.

        If augment is False, then original sentences are returned until end of file is reached. Useful if corpus is large and you cannot load the whole data into memory.

        Else if augment is True, you can keep cycling through the dataset generating new augmented versions of the sentences on each cycle.
        """

        # Returning original sentences as they are if self.augment is False.
        if not self.augment:
            return next(self.src_input_file).rstrip('\n'), next(self.tgt_input_file).rstrip('\n')

        # Augmenting current document.
        src_doc = next(self.src_input_file)
        augmented_src_doc = synonym_aug(src_doc, self.src_pipeline, self.src_net, self.p)
        tgt_doc = next(self.tgt_input_file)
        augmented_tgt_doc = synonym_aug(tgt_doc, self.tgt_pipeline, self.tgt_net, self.p)

        return augmented_src_doc, augmented_tgt_doc

    def __len__(self):
        return self.doc_count