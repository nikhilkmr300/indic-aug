import warnings

import numpy as np
import pandas as pd
from scipy.special import softmax
import stanza

from .globals import VALID_AUG_MODES, ERRORS, PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN, BLANK_TOKEN
from .utils import stanza2list

class DepParseTree:
    """Represents a dependency parsing tree."""

    class Root:
        """Representation of <root> as class with static attribute text for compatibility with ``stanza.models.common.doc.Word``."""

        text = '<root>'

    def __init__(self, sent):
        """Constructor method.

        :param sent: Sentence for which dependency parse tree is to be created. Do NOT pass a ``str``.
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
        """Calculates score of node in dependency parse tree in accordance with :cite:t:`duan2020syntax`.

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
        :param fillcolor: Fill color of node, defaults to 'mediumorchid'. Allowed values are X11 color strings.
        :type fillcolor: str
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

        :param idx: Index for array ``self.words``.
        :type idx: int
        """

        return self.words[idx].text

    def depths2list(self):
        """Returns depths of all nodes in tree as a list. Each element in the list is itself a list in the format [<word_index>, <word>, <depth>].
        """

        return [[idx, self.index2word(idx), self.depths[idx]] for idx in range(len(self.words))]

    def qprobs2list(self):
        """Returns q-probabilities (refer :cite:t:`duan2020syntax`) of all nodes in tree as a list. Each element in the list is itself a list in the format [<word_index>, <word>, <q_probability>].
        """

        return [[idx, self.index2word(idx), self.q_probs[idx]] for idx in range(len(self.words))]

    def softmax2list(self):
        """Returns softmax output (refer :cite:t:`duan2020syntax`) of all nodes in tree as a list. Each element in the list is itself a list in the format [<word_index>, <word>, <softmax_output>].
        """

        return [[idx, self.index2word(idx), self.softmaxed_probs[idx]] for idx in range(len(self.words))]

    def scores2list(self):
        """Returns score (refer :cite:t:`duan2020syntax`) of all nodes in tree as a list. Each element in the list is itself a list in the format [<word_index>, <word>, <score>].
        """

        return [[idx, self.index2word(idx), self.scores[idx]] for idx in range(len(self.words))]

def closest_freq(word, freq_dict):
    """Returns the word in ``freq_dict`` that has the closest frequency to that of ``word``.

    :param word: Word whose closest frequency word is to be found.
    :type word: str
    :param freq_dict: Word to frequency mapping as returned by ``vocab.freq2dict_vocab``.
    :type freq_dict: dict

    :return: Word with closest frequency to that of ``word``.
    :rtype: str
    """

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

def depparse_aug(sent, mode, alpha, freq_dict=None):
    """Performs dependency parsing augmentation (refer :cite:t:`duan2020syntax`) on a sentences.

    ``freq_dict`` is required only if ``mode`` is 'replace', it is ignored otherwise.

    :param sent: Sentence to be augmented. Do NOT pass a str.
    :type sent: ``stanza.models.common.doc.Sentence``
    :param mode: Action to perform after scores are extracted. Valid values are given in globals.py.
    :type mode: str
    :param alpha: Same as for ``DepParseTree.score``.
    :type alpha: float
    :param freq_dict: Dictionary of word-frequency pairs as returned by ``vocab.score2freq_vocab``, defaults to None.
    :type freq_dict: dict, optional
    """

    if mode == 'replace' and freq_dict is None:
        raise ValueError(f'Passing freq_dict is compulsory with mode=\'replace\'.')

    tree = DepParseTree(sent)                   # Dependency parse tree for sentence.
    tree.score(alpha=alpha)

    # Converting stanza sentence to simple list as we no longer need the extra stuff.
    sent = stanza2list(sent)

    scores = tree.scores[1:]                    # Dropping score related to root as it is not required.
    # Sanity check
    assert len(scores) == len(sent)             # After removing score for <root>, lengths should match.

    for idx, word in enumerate(sent):
        # Not replacing special tokens.
        if word in {UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, BLANK_TOKEN}:
            continue

        # CHECK: Generate a new random number for each iteration or use a common one initialized outside the for loop?
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