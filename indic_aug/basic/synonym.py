import logging
from pathlib import Path
import random
import re
import textwrap

import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pyiwn

from ..globals import Augmentor, ERRORS, SENTENCE_DELIMS
from ..vocab import Vocab
from ..log import logger, NUM_LOGGER_DASHES
from ..utils import cyclic_read, path2lang, line_count, doc2words, load_stanza_pipeline

pyiwn.logger.setLevel(logging.WARNING)
logging.getLogger('numexpr.utils').setLevel(logging.WARN)

def convert_pos(stanza_pos, to):
    """Converts Universal Dependencies parts of speech to WordNet/IndoWordNet
    parts of speech.

    :param stanza_pos: POS defined by Universal Dependencies.
    :type stanza_pos: str
    :param to: POS to convert to, can be 'wn' for WordNet and 'iwn' for
        IndoWordNet.
    :type to: str

    :return: POS in WordNet/IndoWordNet format.
    :rtype: str if ``to`` is 'wn', ``pyiwn.iwn.PosTag`` if ``to`` is 'iwn'
    """

    try:
        wn.ADJ
    except LookupError as e:
        logger.info('Could not find nltk resource \'wordnet\'. Downloading resource \'wordnet\'...')
        nltk.download('wordnet')

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

def find_synonyms(word, pipeline, net):
    """Returns a list of synonyms of ``word``.

    :param word: Word whose synonym is required.
    :type word: str
    :param pipeline: ``stanza.Pipeline`` corresponding to language of word.
    :type pipeline: ``stanza.Pipeline``
    :param net: Thesaurus (word net) to use. Use nltk.corpus.wordnet for English
        and pyiwn.iwn.IndoWordNet() for Indian languages.
    :type net: ``nltk.corpus.reader.wordnet.WordNetCorpusReader`` for English
        and ``pyiwn.iwn.IndoWordNet`` for Indian languages.

    :return: List of synonyms for ``word``.
    :rtype: list
    """

    if len(word.split(' ')) != 1:
        raise ValueError(f'Parameter \'word\' to find_synonyms must be a str with no spaces.')

    lang = pipeline.lang

    # POS of word whose synonyms are required in Universal Dependencies format,
    # used by stanza.
    stanza_pos = pipeline(word).sentences[0].words[0].upos
    # POS of word whose synonyms are required in WordNet or IndoWordNet format,
    # used by nltk/pyiwn.
    converted_pos = convert_pos(stanza_pos, 'wn') if lang == 'en' else convert_pos(stanza_pos, 'iwn')

    if converted_pos is None:
        # If converted_pos is None, unknown POS, nltk/pyiwn cannot handle,
        # returning empty list.
        return list()

    try:
        # Synonym sets corresponding to `word` and have the same POS as `word`.
        synsets = net.synsets(word, pos=converted_pos)
    except KeyError as e:
        # If word does not exist in IndoWordNet, returning empty list.
        return list()

    synonyms = list()

    # Extracting synonyms which have the same POS as input word.
    for synset in synsets:
        synonyms.extend(synset.lemma_names())

    # Removing occurrences of word itself from its list of synonyms.
    synonyms = [synonym for synonym in synonyms if synonym != word]

    return synonyms

def synonym_aug(doc, pipeline, net, vocab, p):
    """Performs augmentation on a document by replacing with synonyms (refer:
    :cite:t:`wei2019eda`).

    :param doc: Document to be augmented.
    :type doc: str
    :param pipeline: Same as for ``find_synonyms``.
    :type pipeline: ``stanza.Pipeline``
    :param net: Same as for ``find_synonyms``.
    :type net: ``nltk.corpus.reader.wordnet.WordNetCorpusReader`` for English
        and ``pyiwn.iwn.IndoWordNet`` for Indian languages.
    :param vocab: List of words in vocabulary.
    :type vocab: list(str)
    :param p: Probability of a word to be replaced by one of its synonyms.
    :type p: float

    :return: Augmented document.
    :rtype: str
    """

    augmented_doc = list()

    lang = pipeline.lang

    for idx, word in enumerate(doc2words(doc, lang)):
        if word in set(re.split('|', SENTENCE_DELIMS)):
            # Not replacing punctuations.
            continue

        if np.random.binomial(1, p):
            logger.info(f'\tSampled word \'{word}\' at index {idx} as a potential candidate for synonym replacement.')
            # Limiting synonym list to words in vocabulary.
            synonyms = [synonym for synonym in find_synonyms(word, pipeline, net) if synonym in vocab]
            if not len(synonyms):
                # If no synonyms found, replace word with itself.
                sampled_synonym = word
                logger.info(f'\t\tNo in-vocabulary synonym found for \'{word}\'.')
            else:
                # Else replace word with a randomly sampled synonym.
                sampled_synonym = random.sample(synonyms, 1)[0]
                logger.info(f'\t\tReplaced \'{word}\' with its WordNet/IndoWordNet synonym \'{sampled_synonym}\'.') 
            augmented_doc.append(sampled_synonym)
        else:
            augmented_doc.append(word)

    return ' '.join(augmented_doc)

class SynonymAugmentor(Augmentor):
    """Class to augment parallel corpora by synonym augmentation technique
    (refer: :cite:t:`wei2019eda`).
    """

    def __init__(self, src_input_path, tgt_input_path, vocab_dir, p, stanza_dir=None, augment=True, random_state=1):
        """Constructor method.

        :param src_input_path: Path to aligned source corpus.
        :type src_input_path: str
        :param tgt_input_path: Path to aligned target corpus, corresponding to
            the above source corpus.
        :type tgt_input_path: str
        :param vocab_dir: As described in the docstring for
            ``indic_aug.vocab.Vocab``.
        :type vocab_dir: str
        :param p: Same as for ``synonym_aug``.
        :type p: float
        :param stanza_dir: Path to directory containing stanza models (the
            default is ``~/stanza_resources`` on doing a ``stanza.download``).
        :type stanza_dir: str
        :param augment: Performs augmentation if ``True``, else returns original
            pair of sentences.
        :type augment: bool
        :param random_state: Seed for the random number generator.
        :type random_state: int
        """

        if line_count(src_input_path) != line_count(tgt_input_path):
            raise RuntimeError(ERRORS['corpus_shape'])
        self.doc_count = line_count(src_input_path)

        random.seed(random_state)                   # synonym_aug uses random from standard library.
        np.random.seed(random_state)

        if stanza_dir is None:
            stanza_dir = str(Path.home() / 'stanza_resources')

        self.src_lang = path2lang(src_input_path)
        self.tgt_lang = path2lang(tgt_input_path)

        self.src_vocab = Vocab.load_vocab(vocab_dir, self.src_lang)
        self.tgt_vocab = Vocab.load_vocab(vocab_dir, self.tgt_lang)

        self.augment = augment

        if self.augment:
            # If augment is True, can perform arbitrary number of augmentations
            # by cycling through all the sentences in the corpus repeatedly.
            self.src_input_file = cyclic_read(src_input_path)
            self.tgt_input_file = cyclic_read(tgt_input_path)
        else:
            # Else does one pass through the corpus and stops.
            self.src_input_file = open(src_input_path)
            self.tgt_input_file = open(tgt_input_path)

        self.p = p

        # Setting up stanza pipelines for source and target languages, for use
        # with synonym_aug.
        self.src_pipeline = load_stanza_pipeline(self.src_lang, stanza_dir=stanza_dir)
        self.tgt_pipeline = load_stanza_pipeline(self.tgt_lang, stanza_dir=stanza_dir)

        # Setting up objects to use WordNet/IndoWordNet.
        self.src_net = wn if self.src_lang == 'en' else pyiwn.IndoWordNet()
        self.tgt_net = wn if self.tgt_lang == 'en' else pyiwn.IndoWordNet()

        logger.info(textwrap.dedent(f'\
            SynonymAugmentor\n\
            \tdoc_count={self.doc_count}\n\
            \tsrc_input_path={src_input_path}\n\
            \ttgt_input_path={tgt_input_path}\n\
            \tsrc_lang={self.src_lang}\n\
            \ttgt_lang={self.tgt_lang}\n\
            \tvocab_dir={vocab_dir}\n\
            \tp={self.p}\n\
            \trandom_state={random_state}\n\
            Note: Words are 0-indexed.'
        ))
        logger.info('-' * NUM_LOGGER_DASHES)

    def __next__(self):
        # Returning original sentences as they are if self.augment is False.
        if not self.augment:
            return next(self.src_input_file).rstrip('\n'), next(self.tgt_input_file).rstrip('\n')

        src_doc = next(self.src_input_file)
        logger.info(f'src_doc: \'{src_doc}\'')

        augmented_src_doc = synonym_aug(src_doc, self.src_pipeline, self.src_net, self.src_vocab, self.p)

        tgt_doc = next(self.tgt_input_file)
        logger.info(f'tgt_doc: \'{tgt_doc}\'')

        augmented_tgt_doc = synonym_aug(tgt_doc, self.tgt_pipeline, self.tgt_net, self.tgt_vocab, self.p)

        logger.info(f'augmented_src_doc: \'{augmented_src_doc}\'')
        logger.info(f'augmented_tgt_doc: \'{augmented_tgt_doc}\'')
        logger.info('-' * NUM_LOGGER_DASHES)

        return augmented_src_doc, augmented_tgt_doc

    def __iter__(self):
        return self

    def __len__(self):
        return self.doc_count