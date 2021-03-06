from pathlib import Path
import textwrap
import warnings

import numpy as np
from scipy.special import softmax

from ..globals import Augmentor, VALID_AUG_MODES, ERRORS, PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, BLANK_TOKEN
from ..log import logger, NUM_LOGGER_DASHES
from ..utils import path2lang, cyclic_read, stanza2list, closest_freq, line_count, load_stanza_pipeline

class DepParseTree:
    """Represents a dependency parsing tree."""

    class Root:
        """Representation of <root> as class with static attribute text for
        compatibility with ``stanza.models.common.doc.Word``.
        """

        text = '<root>'

    def __init__(self, sent):
        """Constructor method.

        :param sent: Sentence for which dependency parse tree is to be created.
            Do NOT pass a `str`.
        :type sent: ``stanza.models.common.doc.Sentence``
        """

        self.words = sent.words                 # Words in sentence
        # Appending a dummy root as stanza uses 1-based indexing for words in sentence.
        self.words = [self.Root] + self.words

        self.depths = np.array([self._get_depth(idx) for idx in range(len(self.words))])

        # Sanity check
        assert len(self.depths) == len(self.words)

        # self.q_probs, self.softmaxed_probs and self.scores are unpopulated until self.score has been called.

    def _get_depth(self, idx):
        """Returns depth of node in dependency parse tree.

        :param idx: Index of word in ``self.words`` whose depth is to be found.
        :type idx: int

        :return: Depth of node.
        :rtype: int
        """

        if not idx:
            # Authors consider depth of root to be 1, so q probability for root is 0.
            return 1

        return 1 + self._get_depth(self.words[idx].head)

    def score(self, alpha=0.1):
        """Calculates score of node in dependency parse tree in accordance with
        :cite:t:`duan2020syntax`.

        :param alpha: Hyperparameter which controls likelihood of changing,
            defaults to 0.1.
        :type alpha: float, optional

        :return: Score of node.
        :rtype: float
        """

        # Calculating q-probabilities as defined in the paper.
        self.q_probs = 1 - 1 / (2. ** (self.depths - 1))

        # Applying softmax to generate the probability distribution.
        self.softmaxed_probs = softmax(self.q_probs)
        # Sanity check, sum of softmax probabilities must be 1.
        assert np.isclose(sum(self.softmaxed_probs), 1)

        # Converting softmax output to scores (note that scores need not sum to 1).
        self.scores = alpha * self.softmaxed_probs * (len(self.words) - 1)

        # Equation given in the paper allows values greater than 1.
        if (self.scores > 1).any():
            warnings.warn(f'Raw score out of range [0, 1]. You might want to consider reducing alpha. Clipping scores to range [0, 1].')
        # Clipping values to range [0, 1] as score is used as a probability.
        self.scores = np.clip(self.scores, 0, 1)

        # Sanity check
        assert len(self.scores) == len(self.depths) == len(self.words)

    def save_tree(self, output_path, shape='box', style='filled', fillcolor='mediumorchid', fontname='Courier', fontcolor='black', edgecolor='black'):
        """Saves dependency parse tree to a GraphViz DOT format file.

        :param output_path: Path to output DOT file.
        :type output_path: str
        :param shape: Shape of boundary box of node, defaults to 'box'. Refer
            GraphViz docs for allowed values.
        :type shape: str
        :param style: Style of node, defaults to 'filled'. Refer GraphViz docs
            for allowed values.
        :type style: str
        :param fillcolor: Fill color of node, defaults to 'mediumorchid'.
            Allowed values are X11 color strings.
        :type fillcolor: str
        :param fontname: Font to use for node labels, defaults to 'Courier'.
            Refer GraphViz docs for allowed values.
        :type fontname: str
        :param fontcolor: Color of font for node labels, defaults to 'black'.
            Allowed values are X11 color strings.
        :type fontcolor: str
        :param edgecolor: Color of edges, defaults to 'black'. Allowed values
            are X11 color strings.
        :type edgecolor: str
        """

        with open(output_path, 'w') as f:
            f.write('digraph G {\n')
            f.write(' '.join([
                f'\tnode',
                f'[shape={shape}',
                f'style={style}',
                f'fillcolor={fillcolor}',
                f'fontname={fontname}',
                f'fontcolor={fontcolor}];\n'
            ]))
            f.write(f'\tedge [color={edgecolor}];\n')
            for word_idx in range(len(self.words)):
                if word_idx == 0:
                    # Root has no parent.
                    continue
                f.write(f'\t{self.words[word_idx].head} -> {word_idx};\n')
            for word_idx in range(len(self.words)):
                f.write(f'\t{word_idx} [label="{word_idx}:{self.index2word(word_idx)}"];\n')
            f.write('}')

    def index2word(self, idx):
        """Returns word corresponding to an index.

        :param idx: Index for array ``self.words``.
        :type idx: int
        """

        return self.words[idx].text

    def depths2list(self):
        """Returns depths of all nodes in tree as a list. Each element in the
        list is itself a list in the format [<word_index>, <word>, <depth>].
        """

        return [[idx, self.index2word(idx), self.depths[idx]] for idx in range(len(self.words))]

    def qprobs2list(self):
        """Returns q-probabilities (refer :cite:t:`duan2020syntax`) of all nodes
        in tree as a list. Each element in the list is itself a list in the
        format [<word_index>, <word>, <q_probability>].
        """

        return [[idx, self.index2word(idx), self.q_probs[idx]] for idx in range(len(self.words))]

    def softmax2list(self):
        """Returns softmax output (refer :cite:t:`duan2020syntax`) of all nodes
        in tree as a list. Each element in the list is itself a list in the
        format [<word_index>, <word>, <softmax_output>].
        """

        return [[idx, self.index2word(idx), self.softmaxed_probs[idx]] for idx in range(len(self.words))]

    def scores2list(self):
        """Returns score (refer :cite:t:`duan2020syntax`) of all nodes in tree
        as a list. Each element in the list is itself a list in the format
        [<word_index>, <word>, <score>].
        """

        return [[idx, self.index2word(idx), self.scores[idx]] for idx in range(len(self.words))]

def depparse_aug(sent, mode, alpha, freq_dict=None):
    """Performs augmentation on one sentence (and NOT a document of sentences,
    as some of the other augmentorss) by dependency parsing (refer
    :cite:t:`duan2020syntax`).

    ``freq_dict`` is required only if ``mode`` is 'replace', it is ignored
    otherwise.

    :param sent: Sentence to be augmented. Do NOT pass a str or a list of str.
    :type sent: ``stanza.models.common.doc.Sentence``
    :param mode: Action to perform after scores are extracted. Valid values are
        given in globals.py.
    :type mode: str
    :param alpha: Same as for ``DepParseTree.score``.
    :type alpha: float
    :param freq_dict: Dictionary of word-frequency pairs as returned by
        ``vocab.score2freq_vocab``, defaults to None.
    :type freq_dict: dict, optional

    :return: Augmented sentence.
    :rtype: str
    """

    if mode == 'replace_freq' and freq_dict is None:
        raise ValueError(ERRORS['freq_dict_compulsory'])

    tree = DepParseTree(sent)                   # Dependency parse tree for sentence.
    tree.score(alpha=alpha)

    # Converting stanza sentence to simple list as we no longer need the extra
    # stuff.
    sent = stanza2list(sent)

    scores = tree.scores[1:]                    # Dropping score related to root as it is not required.
    # Sanity check
    assert len(scores) == len(sent)             # After removing score for <root>, lengths should match.

    for idx, word in enumerate(sent):
        # Not replacing special tokens.
        if word in {UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, BLANK_TOKEN}:
            continue

        if np.random.binomial(1, scores[idx]):
            if mode == 'blank':
                # Blanking with probability scores[idx].
                sent[idx] = BLANK_TOKEN
                logger.info(f'\tBlanked word \'{word}\' at index {idx}.')
            elif mode == 'dropout':
                # Dropping word with probability scores[idx].
                sent[idx] = ''
                logger.info(f'\tDropped word \'{word}\' at index {idx}.')
            elif mode == 'replace_freq':
                # Replacing word with probability scores[idx].
                replacement = closest_freq(word, freq_dict)
                sent[idx] = replacement
                logger.info(f'\tReplaced word \'{word}\' at index {idx} with \'{replacement}\'.')
            else:
                raise ValueError(f'Invalid value of parameter \'mode\'. Valid values for parameter \'mode\' to depparse_aug are values in {*VALID_AUG_MODES["depparse"],}.')

    # Stripping any extra whitespace that has been introduced by dropout.
    sent = list(filter(lambda x: x != '', sent))

    return ' '.join(sent)

class DepParseAugmentor(Augmentor):
    """Class to augment parallel corpora by dependency parsing technique (refer:
    :cite:t:`duan2020syntax`).
    """

    def __init__(self, src_input_path, tgt_input_path, mode, alpha, src_freq_dict=None, tgt_freq_dict=None, stanza_dir=str(Path.home() / 'stanza_resources'), augment=True, random_state=1):
        """Constructor method.

        :param src_input_path: Path to aligned source corpus.
        :type src_input_path: str
        :param tgt_input_path: Path to aligned target corpus, corresponding to
            the above source corpus.
        :type tgt_input_path: str
        :param mode: Same as for ``depparse_aug``.
        :type mode: str
        :param alpha: Same as for ``DepParseTree``.
        :type alpha: float
        :param src_freq_dict: Same as for ``freq_dict`` in
            ``transforms.depparse_aug``.
        :type src_freq_dict: dict
        :param tgt_freq_dict: Same as for ``freq_dict`` in
            ``transforms.depparse_aug``.
        :type tgt_freq_dict: dict
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

        np.random.seed(random_state)

        src_lang = path2lang(src_input_path)
        tgt_lang = path2lang(tgt_input_path)

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

        self.mode = mode
        self.alpha = alpha
        self.src_freq_dict = src_freq_dict
        self.tgt_freq_dict = tgt_freq_dict

        # Loading stanza pipeline for source and target languages to convert
        # string to stanza.models.common.doc.Sentence.
        self.src_pipeline = load_stanza_pipeline(src_lang, stanza_dir=stanza_dir)
        self.tgt_pipeline = load_stanza_pipeline(tgt_lang, stanza_dir=stanza_dir)

        logger.info(textwrap.dedent(f'\
            DepParseAugmentor\n\
            \tdoc_count={self.doc_count}\n\
            \tsrc_input_path={src_input_path}\n\
            \ttgt_input_path={tgt_input_path}\n\
            \tsrc_lang={src_lang}\n\
            \ttgt_lang={tgt_lang}\n\
            \tmode={self.mode}\n\
            \talpha={self.alpha}\n\
            \trandom_state={random_state}\n\
            Note: Words are 0-indexed.'
        ))
        logger.info('-' * NUM_LOGGER_DASHES)

    def __next__(self):
        # Returning original sentences as they are if self.augment is False.
        if not self.augment:
            return next(self.src_input_file).rstrip('\n'), next(self.tgt_input_file).rstrip('\n')

        src_doc = next(self.src_input_file).rstrip('\n')
        logger.info(f'src_doc: \'{src_doc}\'')

        src_doc = self.src_pipeline(src_doc)
        augmented_src_doc = list()
        for src_sent in src_doc.sentences:
            augmented_src_doc.append(depparse_aug(src_sent, self.mode, self.alpha, self.src_freq_dict))

        tgt_doc = next(self.tgt_input_file).rstrip('\n')
        logger.info(f'tgt_doc: \'{tgt_doc}\'')

        tgt_doc = self.tgt_pipeline(tgt_doc)
        augmented_tgt_doc = list()
        for tgt_sent in tgt_doc.sentences:
            augmented_tgt_doc.append(depparse_aug(tgt_sent, self.mode, self.alpha, self.tgt_freq_dict))

        # Joining all sentences in document.
        augmented_src_doc = ' '.join(augmented_src_doc)
        augmented_tgt_doc = ' '.join(augmented_tgt_doc)

        logger.info(f'augmented_src_doc: \'{augmented_src_doc}\'')
        logger.info(f'augmented_tgt_doc: \'{augmented_tgt_doc}\'')
        logger.info('-' * NUM_LOGGER_DASHES)

        return augmented_src_doc, augmented_tgt_doc

    def __iter__(self):
        return self

    def __len__(self):
        return self.doc_count