import numpy as np
from scipy.special import softmax
import stanza

class DepParseTree:
    """Represents a dependency parsing tree."""

    class Root:
        """Representation of <root> as class with static attribute text for compatibility with `stanza.models.common.doc.Word`."""

        text = '<root>'

    def __init__(self, sent):
        """Constructor method."""

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
            for word in self.words:
                this_text = word.text
                if this_text == '<root>':
                    continue
                parent_text = self.words[word.head].text
                f.write(f'\t{parent_text} -> {this_text};\n')
            for idx in range(len(self.words)):
                f.write(f'\t{self.index2word(idx)} [label="{idx}:{self.index2word(idx)}"];\n')
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