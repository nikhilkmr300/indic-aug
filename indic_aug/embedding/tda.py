from ..globals import Augmentor, SENTENCE_DELIMS, ERRORS, PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, BLANK_TOKEN
from ..lm import load
from ..vocab import read_bottomn_vocab
from ..utils import line_count, path2lang, cyclic_read

def extract_rare_words(model_path, vocab_path, word_count):
    """Wrapper around vocab.read_bottomn_vocab. The difference is that this
    function removes words that we do not want replaced according to the
    translation data augmentation (TDA) algorithm (refer:
    :cite:t:`fadaee2017data`).

    :param model_path: Path to \*.model file compatible with ``sentencepiece``.
    :type model_path: str
    :param vocab_path: Path to \*.vocab file compatible with ``sentencepiece``,
        corresponding to model at ``model_path``.
    :type vocab_path: str
    :param word_count: Number of words to include in the set of targeted words.
        Note that the actual number of words returned might be slightly less 
        than the specified ``word_count`` as special tokens such as SOS, EOS, 
        UNK, etc. are removed.
    :type word_count: int

    :return: Output of ``vocab.read_bottomn_vocab`` minus the special tokens we
        do not want replaced.
    :rtype: list
    """

    rare_words = list(read_bottomn_vocab(model_path, vocab_path, word_count).keys())

    # Removing special words that we don't want to be replaced.
    words_to_remove = [sentence_delim.strip('\\') for sentence_delim in SENTENCE_DELIMS.split('|')]
    words_to_remove.extend([PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, BLANK_TOKEN])

    for word_to_remove in words_to_remove:
        if word_to_remove in rare_words:
            rare_words.remove(word_to_remove)

    return rare_words

def suggest_substitution(left_context, lang_model, rare_words):
    suggested_word = lang_model.predict(left_context)

    if suggested_word in rare_words:
        return suggested_word

    return None

class ParallelAugmentor(Augmentor):
    """Class to augment parallel corpora by translation data augmentation
    (parallel) technique (refer: :cite:t:`fadaee2017data`).
    """

    def __init__(self, src_input_path, tgt_input_path, aligner, augment=True):
        if line_count(src_input_path) != line_count(tgt_input_path):
            raise RuntimeError(ERRORS['corpus_shape'])
        self.doc_count = line_count(src_input_path)

        src_lang = path2lang(src_input_path)
        tgt_lang = path2lang(tgt_input_path)

        self.src_lm = load(src_lang)
        self.tgt_lm = load(tgt_lang)

        self.aligner = aligner

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

    def __next__(self):
        # Returning original sentences as they are if self.augment is False.
        if not self.augment:
            return next(self.src_input_file).rstrip('\n'), next(self.tgt_input_file).rstrip('\n')

        src_doc = next(self.src_input_file).rstrip('\n')
        tgt_doc = next(self.tgt_input_file).rstrip('\n')

        # Placeholder list to hold augmented document, will join all sentences
        # in document before returning.
        augmented_src_doc = list()
        augmented_tgt_doc = list()



    def __iter__(self):
        pass

    def __len__(self):
        pass