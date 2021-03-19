import warnings

import numpy as np
import pandas as pd
from scipy.special import softmax
import stanza

from .globals import VALID_AUG_MODES, ERRORS, SOS_TOKEN, EOS_TOKEN, BLANK_TOKEN
from .utils import stanza2list

class DepParseTree:
    """Represents a dependency parsing tree."""

    class Root:
        """Representation of <root> as class with static attribute text for compatibility with `stanza.models.common.doc.Word`."""

        text = '<root>'

    def __init__(self, sent):
        """Constructor method.

        :param sent: Sentence for which dependency parse tree is to be created. Do NOT pass a `str`.
        :type sent: `stanza.models.common.doc.Sentence`
        """

        self.words = sent.words                 # Words in sentence
        # Appending a dummy root as stanza uses 1-based indexing for words in sentence.
        self.words = [self.Root] + self.words

        self.depths = np.array([self._get_depth(idx) for idx in range(1, len(self.words))])
        # Prepending a 0 depth for root.
        self.depths = np.insert(self.depths, 0, 0)

        # Sanity check
        assert len(self.depths) == len(self.words)

        # Unpopulated until self.score has been called.
        self.q_probs = None
        self.softmaxed_probs = None
        self.scores = None

    def _get_depth(self, idx):
        """Returns depth of node in dependency parse tree.

        :param idx: Index of word in `self.words` whose depth is to be found.
        :type idx: int

        :return: Depth of node.
        :rtype: int
        """

        if not idx:
            return 0

        return 1 + self._get_depth(self.words[idx].head)

    def score(self, alpha=0.1):
        """Calculates score of node in dependency parse tree in accordance with :cite:p:`duan2020syntax`.

        :param alpha: Hyperparameter which controls likelihood of changing, defaults to 0.1.
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
        :param shape: Shape of boundary box of node, defaults to 'box'. Refer GraphViz docs for allowed values.
        :type shape: str
        :param style: Style of node, defaults to 'filled'. Refer GraphViz docs for allowed values.
        :type style: str
        :param fillcolor: Fill color of node, defaults to 'mediumorchid'.
        :type fillcolor: str. Allowed values are X11 color strings.
        :param fontname: Font to use for node labels, defaults to 'Courier'. Refer GraphViz docs for allowed values.
        :type fontname: str
        :param fontcolor: Color of font for node labels, defaults to 'black'. Allowed values are X11 color strings.
        :type fontcolor: str
        :param edgecolor: Color of edges, defaults to 'black'. Allowed values are X11 color strings.
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

        :param idx: Index for array `self.words`.
        :type idx: int
        """

        return self.words[idx].text

    def depths2list(self):
        """Returns depths of all nodes in tree as a list. Each element in the list is itself a list in the format [<word_index>, <word>, <depth>].
        """

        return [[idx, self.index2word(idx), self.depths[idx]] for idx in range(len(self.words))]

    def qprobs2list(self):
        """Returns q-probabilities (refer :cite:p:`duan2020syntax`) of all nodes in tree as a list. Each element in the list is itself a list in the format [<word_index>, <word>, <q_probability>].
        """

        return [[idx, self.index2word(idx), self.q_probs[idx]] for idx in range(len(self.words))]

    def softmax2list(self):
        """Returns softmax output (refer :cite:p:`duan2020syntax`) of all nodes in tree as a list. Each element in the list is itself a list in the format [<word_index>, <word>, <softmax_output>].
        """

        return [[idx, self.index2word(idx), self.softmaxed_probs[idx]] for idx in range(len(self.words))]

    def scores2list(self):
        """Returns score (refer :cite:p:`duan2020syntax`) of all nodes in tree as a list. Each element in the list is itself a list in the format [<word_index>, <word>, <score>].
        """

        return [[idx, self.index2word(idx), self.scores[idx]] for idx in range(len(self.words))]

def closest_freq(word, freq_dict):
    if not word in freq_dict.keys():
        raise ERRORS['word_not_found']

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

def depparse_aug(src_sent, tgt_sent, mode, alpha, src_freq_dict=None, tgt_freq_dict=None):
    """Performs dependency parsing augmentation (refer :cite:p`duan2020syntax) on source and target sentences.

    `src_freq_dict` and `tgt_freq_dict` are required only if `mode` is 'replace', ignored otherwise.

    :param src_sent: Source sentence to be augmented. Do NOT pass a str.
    :type src_sent: `stanza.models.common.doc.Sentence`
    :param tgt_sent: Corresponding target sentence to be augmented. Do NOT pass a str.
    :type tgt_sent: `stanza.models.common.doc.Sentence`
    :param mode: Action to perform after scores are extracted. Valid values are given in globals.py.
    :type mode: str
    :param alpha: Same as for `DepParseTree.score`.
    :type alpha: float
    :param src_freq_dict: Dictionary of source word-frequency pairs as returned by `vocab.score2freq_vocab`, defaults to None.
    :type src_freq_dict: dict, optional
    :param tgt_freq_dict: Dictionary of target word-frequency pairs as returned by `vocab.score2freq_vocab`, defaults to None.
    :type tgt_freq_dict: dict, optional
    """

    def apply_mode(sent, scores, mode, freq_dict=None):
        """Performs the actual step of blank/dropout/replace.

        Again, `freq_dict` is required only when `mode` is 'replace'.

        :param sent: Sentence to be augmented. Pass a list of tokens as returned by `utils.stanza2list`, NOT `stanza.models.common.Sentence`.
        :type sent: list
        :param scores: Array of scores corresponding to each word.
        :type scores: `arraylike`
        :param mode: One of 'blank', 'dropout' and 'replace'.
        :type mode: str
        :param freq_dict: Dictionary of word-frequency pairs as returned by `vocab.score2freq_vocab`, defaults to None.
        :type freq_dict: dict

        :return: Augmented sentences as a tuple of str.
        :rtype: tuple
        """

        for idx, word in enumerate(sent):
            # Not replacing start of sentence and end of sentence tokens.
            if word in {SOS_TOKEN, EOS_TOKEN}:
                continue

            if np.random.uniform() < scores[idx]:
                if mode == 'blank':
                    # Blanking with probability src_scores[idx].
                    sent[idx] = BLANK_TOKEN
                elif mode == 'dropout':
                    # Dropping word with probability src_scores[idx].
                    sent[idx] = ''
                elif mode == 'replace':
                    # Replacing word with probability src_scores[idx].
                    sent[idx] = closest_freq(word, freq_dict)

        # Stripping any extra whitespace that has been introduced by dropout.
        sent = list(filter(lambda x: x != '', sent))

        return ' '.join(sent)

    if mode == 'replace' and (src_freq_dict is None or tgt_freq_dict is None):
        raise ValueError(f'Passing freq_dict is compulsory with mode=\'replace\'.')

    src_tree = DepParseTree(src_sent)               # Dependency parse tree for source sentence.
    src_tree.score(alpha=alpha)
    tgt_tree = DepParseTree(tgt_sent)               # Dependency parse tree for target sentence.
    tgt_tree.score(alpha=alpha)

    # Converting stanza sentences to simple list as we no longer need the extra stuff.
    src_sent = stanza2list(src_sent)
    tgt_sent = stanza2list(tgt_sent)

    src_scores = src_tree.scores[1:]                # Dropping score related to root as it is not required.
    # Sanity check
    assert len(src_scores) == len(src_sent)         # After removing score for <root>, lengths should match.

    tgt_scores = tgt_tree.scores[1:]                # Dropping score related to root as it is not required.
    # Sanity check
    assert len(tgt_scores) == len(tgt_sent)         # After removing score for <root>, lengths should match.

    src_sent = apply_mode(src_sent, src_scores, mode, src_freq_dict)
    tgt_sent = apply_mode(tgt_sent, tgt_scores, mode, tgt_freq_dict)

    return src_sent, tgt_sent