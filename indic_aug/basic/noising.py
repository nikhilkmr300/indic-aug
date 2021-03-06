import re
import textwrap

import numpy as np

from ..globals import Augmentor, VALID_AUG_MODES, BLANK_TOKEN, ERRORS, SENTENCE_DELIMS
from ..log import logger, NUM_LOGGER_DASHES
from ..utils import cyclic_read, path2lang, line_count, doc2sents, sent2words, doc2words

def count_unigrams(path):
    """Returns the count of all the unigrams that appear in the corpus at
    ``path``. Corpus at ``path`` must be formatted so that each line contains a
    document, and each document may contain one or more sentences terminated
    either by the fullstop character (\u002e) or the poorna virama character
    (\u0964). Each sentence is made of space-separated tokens.

    :param path: Path to input corpus.
    :type path: str

    :return: Map of unigram to number of occurrences of that unigram.
    :rtype: dict
    """

    unigram_counts = dict()

    lang = path2lang(path)

    with open(path, 'r') as f:
        for doc in f:
            for sent in doc2sents(doc, lang):
                for word in sent2words(sent, lang):
                    if word == '':
                        # '' is end of sentence, not counting it as unigram.
                        continue
                    if word in unigram_counts.keys():
                        unigram_counts[word] += 1
                    else:
                        unigram_counts[word] = 1

    return unigram_counts


def sample_word(unigram_counts):
    """Samples words from a probability distribution of words.

    :param unigram_counts: Map of words (str) to their number of occurrences
        (int), as returned by ``count_unigrams``.
    :type: dict

    :return: Word randomly sampled according to the probability distribution of
        counts.
    :rtype: str
    """

    total_count = sum(unigram_counts.values())     # Total number of words.
    # Converting counts to frequencies in range [0, 1].
    words2probs = {word: word_count / total_count for word, word_count in unigram_counts.items()}
    # Converting word_probs to a list of [key, val] so that order is maintained.
    words2probs = [[word, word_count / total_count] for word, word_count in unigram_counts.items()]

    # Extracting word probabilities to pass as probabilities to numpy.random.choice.
    word_probs = (lambda xs: [x[1] for x in xs])(words2probs)

    # Sampling word index probability distribution word_probs.
    word_idx = np.random.choice(np.arange(len(word_probs)), p=word_probs)

    # Returning the word corresponding to word_idx.
    return words2probs[word_idx][0]

def count_bigrams(path):
    """Returns the count of all the bigrams that appear in the corpus at
    ``path``. Corpus at ``path`` must be formatted so that each line contains a
    document, and each document may contain one or more sentences terminated
    either by the fullstop character (\u002e) or the poorna virama character
    (\u0964). Each sentence is made of space-separated tokens.

    :param path: Path to input corpus.
    :type path: str

    :return: Map of bigram (represented as a 2-tuple) to number of occurrences
        of that bigram.
    :rtype: dict
    """

    bigram_counts = dict()

    lang = path2lang(path)

    with open(path, 'r') as f:
        for doc in f:
            for sent in doc2sents(doc, lang):
                words = sent2words(sent, lang)
                for idx in range(len(words) - 1):
                    if (words[idx], words[idx + 1]) in bigram_counts.keys():
                        bigram_counts[(words[idx], words[idx + 1])] += 1
                    else:
                        bigram_counts[(words[idx], words[idx + 1])] = 1

    return bigram_counts

def find_next_sets(bigram_counts):
    """Returns the next set of first unigrams occurring in ``bigram_counts``.

    Defining next set of a word as the set of second unigrams where that word is
    the first unigram. For example, say we have the bigrams ('hello', 'world'),
    ('hello', 'alice') and ('hello', 'bob'), then the next set of 'hello' would
    be {'world', 'alice', 'bob'}.

    Return value of this function is a dictionary of first unigram to its next
    set pairs.

    :param bigram_counts: Dictionary of bigram (2-tuple) to bigram count pairs,
        as returned by ``count_bigrams``.
    :type bigram_counts: dict

    :return: Next sets of all first unigrams.
    :rtype: dict
    """

    next_sets = dict()

    for first_word, second_word in bigram_counts.keys():
        if first_word == '' or second_word == '':
            # '' represents end of sentence, don't count this bigram.
            continue
        if not first_word in next_sets.keys():
            next_sets[first_word] = {second_word}
        else:
            next_sets[first_word].add(second_word)

    return next_sets

def find_prev_sets(bigram_counts):
    """Returns the prev set of second unigrams occurring in ``bigram_counts``.

    Defining prev set of a word as the set of first unigrams where that word is
    the second unigram.
    For example, say we have the bigrams ('red', 'flag'), ('green', 'flag'),
    ('white', 'flag'), then the prev set of 'flag' would be {'red', 'green',
    'flag'}.

    Return value of this function is a dictionary of first unigram to its next
    set pairs.

    :param bigram_counts: Dictionary of bigram (2-tuple) to bigram count pairs,
        as returned by ``count_bigrams``.
    :type bigram_counts: dict

    :return: Next sets of all first unigrams.
    :rtype: dict
    """

    prev_sets = dict()

    for first_word, second_word in bigram_counts.keys():
        if first_word == '' or second_word == '':
            # '' represents end of sentence, don't count this bigram.
            continue
        if not second_word in prev_sets.keys():
            prev_sets[second_word] = {first_word}
        else:
            prev_sets[second_word].add(first_word)

    return prev_sets

def noising_aug(doc, mode, gamma0, unigram_counts, lang, next_sets, prev_sets=None):
    """Performs augmentation on a document by blanking/replacement (refer:
    :cite:t:`xie2017data`).

    Supports all the four modes specified in the paper:
        * `blank`: Replaces token with ``BLANK_TOKEN`` with noising probability
            ``gamma0``.
        * `replace`: Replaces token with another token from the unigram
            distribution with noising probability ``gamma0``.
        * `absolute_discount`: Replaces token with another token from the
            unigram distribution with absoute discounted probability obtained 
            from ``gamma0``.
        * `kneser_ney`: Replaces token with another token from its prev set with
            absoute discounted probability obtained from ``gamma0``, analogous 
            to Kneser-Ney smoothing.

    :param doc: Document to be augmented.
    :type doc: str
    :param mode: One of the modes described above.
    :type mode: str
    :param gamma0: Noising probability. Values are clipped to range [0, 1].
    :type gamma0: float
    :param lang: ISO 639-1 language code of ``doc``.
    :type lang: str
    :param next_sets: Next sets of all tokens in corpus, as returned by
        `find_next_sets`.
    :type next_sets: dict
    :param prev_sets: Prev sets of all tokens in corpus, as returned by
        `find_next_sets`, optional. Required if ``mode`` is 'kneser_ney', else
        ignored.
    :type prev_sets: dict

    :return: Augmented document.
    :rtype: str
    """

    augmented_doc = list()

    for idx, word in enumerate(doc2words(doc, lang)):
        if word in set(re.split('|', SENTENCE_DELIMS)):
            # Not noising punctuations.
            continue

        if mode == 'blank':
            if np.random.binomial(1, gamma0):
                # Blanking with probability gamma0.
                augmented_doc.append(BLANK_TOKEN)
                logger.info(f'\tBlanked word \'{word}\' at index {idx}.')
            else:
                augmented_doc.append(word)

        elif mode == 'replace':
            if np.random.binomial(1, gamma0):
                sampled_word = sample_word(unigram_counts)
                # Replacing from unigram distribution with probability gamma0.
                augmented_doc.append(sampled_word)
                logger.info(f'\tReplaced word \'{word}\' at index {idx} with \'{sampled_word}\'.')
            else:
                augmented_doc.append(word)

        elif mode == 'absolute_discount':
            numer = len(next_sets[word]) if word in next_sets.keys() else 0
            denom = unigram_counts[word]
            gammaAD = gamma0 * numer / denom     # Absolute discounted gamma, refer Xie's paper.

            if np.random.binomial(1, gammaAD):
                sampled_word = sample_word(unigram_counts)
                # Replacing from unigram distribution with probability gammaAD.
                augmented_doc.append(sampled_word)
                logger.info(f'\tReplaced word \'{word}\' with \'{sampled_word}\'.')
            else:
                augmented_doc.append(word)

        elif mode == 'kneser_ney':
            if prev_sets is None:
                raise ValueError(ERRORS['prev_set_compulsory'])

            numer = len(next_sets[word]) if word in next_sets.keys() else 0
            denom = unigram_counts[word]
            gammaAD = gamma0 * numer / denom     # Absolute discounted gamma, refer Xie's paper.

            # Unigram counts of words in prev set of word, calling it Kneser-Ney
            # distribution.
            kneser_ney_distr = dict()
            if word in prev_sets.keys():
                for first_word in prev_sets[word]:
                    kneser_ney_distr[first_word] = unigram_counts[first_word]

            if np.random.binomial(1, gammaAD):
                # Replacing from Kneser-Ney distribution with probability
                # gammaAD.
                if not len(kneser_ney_distr):
                    # If no words in prev set, replace word with itself.
                    sampled_word = word
                else:
                    # Otherwise sample from Kneser-Ney distribution.
                    sampled_word = sample_word(kneser_ney_distr)
                    logger.info(f'\tReplaced word \'{word}\' [gammaAD={gammaAD}] with \'{sampled_word}\'.')
                augmented_doc.append(sampled_word)
            else:
                augmented_doc.append(word)

        else:
            raise ValueError(f'Invalid value of parameter \'mode\'. Valid values for parameter \'mode\' to noising_aug are values in {*VALID_AUG_MODES["noising"],}.')

    return ' '.join(augmented_doc)

class NoisingAugmentor(Augmentor):
    """Class to augment parallel corpora by blanking/replacement (refer: :cite:t:`xie2017data`)."""

    def __init__(self, src_input_path, tgt_input_path, mode, gamma0, augment=True, random_state=1):
        """Constructor method.

        :param src_input_path: Path to aligned source corpus.
        :type src_input_path: str
        :param tgt_input_path: Path to aligned target corpus, corresponding to
            the above source corpus.
        :type tgt_input_path: str
        :param mode: Same as for ``noising_aug``.
        :type mode: str
        :param gamma0: Same as for ``noising_aug``.
        :type gamma0: float
        :param augment: Performs augmentation if ``True``, else returns original
            pair of sentences.
        :type augment: bool
        :param random_state: Seed for the random number generator.
        :type random_state: int
        """

        if line_count(src_input_path) != line_count(tgt_input_path):
            raise RuntimeError(ERRORS['corpus_shape'])
        self.doc_count = line_count(src_input_path)

        np.random.seed(random_state)

        self.src_lang = path2lang(src_input_path)
        self.tgt_lang = path2lang(tgt_input_path)

        self.mode = mode
        if not self.mode in VALID_AUG_MODES['noising']:
            raise ValueError(f'Invalid value of parameter \'mode\'. Valid values for parameter \'mode\' to depparse_aug are values in {*VALID_AUG_MODES["noising"],}.')

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

        self.gamma0 = gamma0

        self.src_unigram_counts = count_unigrams(src_input_path)
        self.tgt_unigram_counts = count_unigrams(tgt_input_path)
        src_bigram_counts = count_bigrams(src_input_path)
        tgt_bigram_counts = count_bigrams(tgt_input_path)

        self.src_next_sets = find_next_sets(src_bigram_counts)
        self.tgt_next_sets = find_next_sets(tgt_bigram_counts)

        if self.mode == 'kneser_ney':
            # Only Kneser-Ney noising requires previous sets.
            self.src_prev_sets = find_prev_sets(src_bigram_counts)
            self.tgt_prev_sets = find_prev_sets(tgt_bigram_counts)
        else:
            self.src_prev_sets = None
            self.tgt_prev_sets = None

        logger.info(textwrap.dedent(f'\
            NoisingAugmentor\n\
            \tdoc_count={self.doc_count}\n\
            \tsrc_input_path={src_input_path}\n\
            \ttgt_input_path={tgt_input_path}\n\
            \tsrc_lang={self.src_lang}\n\
            \ttgt_lang={self.tgt_lang}\n\
            \tgamma0={self.gamma0}\n\
            \tmode={self.mode}\n\
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

        augmented_src_doc = noising_aug(src_doc, self.mode, self.gamma0, self.src_unigram_counts, self.src_lang, self.src_next_sets, self.src_prev_sets)

        tgt_doc = next(self.tgt_input_file)
        logger.info(f'tgt_doc: \'{tgt_doc}\'')

        augmented_tgt_doc = noising_aug(tgt_doc, self.mode, self.gamma0, self.tgt_unigram_counts, self.tgt_lang, self.tgt_next_sets, self.tgt_prev_sets)

        logger.info(f'augmented_src_doc: \'{augmented_src_doc}\'')
        logger.info(f'augmented_tgt_doc: \'{augmented_tgt_doc}\'')
        logger.info('-' * NUM_LOGGER_DASHES)

        return augmented_src_doc, augmented_tgt_doc

    def __iter__(self):
        return self

    def __len__(self):
        return self.doc_count