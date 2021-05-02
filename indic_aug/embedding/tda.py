import os

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

class ParallelAugmentor(Augmentor):
    """Class to augment parallel corpora by translation data augmentation
    (parallel) technique (refer: :cite:t:`fadaee2017data`).
    """

    def __init__(self, src_input_path, tgt_input_path, aligner, rare_word_count, src_model_path, src_vocab_path, augment=True):
        if line_count(src_input_path) != line_count(tgt_input_path):
            raise RuntimeError(ERRORS['corpus_shape'])
        self.doc_count = line_count(src_input_path)

        src_lang = path2lang(src_input_path)
        tgt_lang = path2lang(tgt_input_path)

        model_lang = os.path.basename(src_model_path)[:2]
        vocab_lang = os.path.basename(src_vocab_path)[:2]

        if model_lang != src_lang or vocab_lang != src_lang:
            raise RuntimeError('src_model_path and src_vocab_path must correspond to lang={src_lang}.')

        self.src_lm = load(src_lang)        # Language model for source language.
        self.tgt_lm = load(tgt_lang)        # Language model for target language.

        self.rare_words = extract_rare_words(src_model_path, src_vocab_path, rare_word_count)
        print(f'Rare words: {self.rare_words}')

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

        src_doc = next(self.src_input_file).rstrip('\n').split(' ')
        tgt_doc = next(self.tgt_input_file).rstrip('\n').split(' ')

        # Placeholder list to hold augmented document, will join all sentences
        # in document before returning.
        augmented_src_doc = list()
        augmented_tgt_doc = list()

        is_modified = False

        for i in range(1, len(src_doc)):
            context = ' '.join(src_doc[:i])         # Analogous to s_1^{i-1}
            lm_pred = self.src_lm.predict(context)  # Analogous to s_{i}'.

            if lm_pred in self.rare_words:
                print(f'Rare word {lm_pred} replacing {src_doc[i]}.')
                augmented_src_doc.append(lm_pred)

                # Finding word corresponding to s_{i} in target sentence.
                alignment = self.aligner.align(' '.join(src_doc), ' '.join(tgt_doc)).alignment

    def __iter__(self):
        pass

    def __len__(self):
        pass