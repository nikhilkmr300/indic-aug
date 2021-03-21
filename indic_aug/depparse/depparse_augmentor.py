import os

import numpy as np
import stanza

from ..transforms import depparse_aug
from ..utils import path2lang, cyclic_read

class DepParseAugmentor:
    """Class to augment parallel corpora according to :cite:t:`duan2020syntax`."""

    def __init__(self, src_input_path, tgt_input_path, mode, alpha, src_freq_dict=None, tgt_freq_dict=None, augment=True, stanza_dir=os.path.join('~', 'stanza_resources'), random_state=1):
        """Constructor method.

        :param src_input_path: Path to aligned source corpus.
        :type src_input_path: str
        :param tgt_input_path: Path to aligned target corpus, corresponding to the above source corpus.
        :type tgt_input_path: str
        :param mode: Same as for ``transforms.depparse_aug``.
        :type mode: str
        :param alpha: Same as for ``transforms.DepParseTree``.
        :type alpha: float
        :param src_freq_dict: Same as for ``freq_dict`` in ``transforms.depparse_aug``.
        :type src_freq_dict: dict
        :param tgt_freq_dict: Same as for ``freq_dict`` in ``transforms.depparse_aug``.
        :type tgt_freq_dict: dict
        :param augment: Performs augmentation if True, else returns original pair of sentences.
        :type augment: bool
        :param stanza_dir: Path to directory containing stanza models (the default is ``~/stanza_resources``) on doing a ``stanza.download``).
        :type stanza_dir: str
        :param random_state: Seed for the random number generator.
        :type random_state: int
        """

        np.random.seed(random_state)

        src_lang = path2lang(src_input_path)
        tgt_lang = path2lang(tgt_input_path)

        self.augment = augment

        if self.augment:
            # If augment is True, can perform arbitrary number of augmentations by cycling through all the sentences in the corpus repeatedly.
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

        # Loading stanza pipeline for source language to convert string to stanza.models.common.doc.Sentence.
        try:
            self.src_pipeline = stanza.Pipeline(src_lang)
        except (stanza.pipeline.core.ResourcesFileNotFoundError, stanza.pipeline.core.LanguageNotDownloadedError) as e:
            print(f'Could not find stanza model at {stanza_dir}. Downloading model...')
            print(f'If you have already downloaded the model, stop this process (Ctrl-C) and pass the path to the model to parameter stanza_dir.')
            stanza.download(src_lang)
            self.src_pipeline = stanza.Pipeline(src_lang)

        # Loading stanza pipeline for target language to convert string to stanza.models.common.doc.Sentence.
        try:
            self.tgt_pipeline = stanza.Pipeline(tgt_lang)
        except (stanza.pipeline.core.ResourcesFileNotFoundError, stanza.pipeline.core.LanguageNotDownloadedError) as e:
            print(f'Could not find stanza model at {stanza_dir}. Downloading model...')
            print(f'If you have already downloaded the model, stop this process (Ctrl-C) and pass the path to the model to parameter stanza_dir.')
            stanza.download(tgt_lang)
            self.tgt_pipeline = stanza.Pipeline(tgt_lang)

    def __next__(self):
        """Returns a pair of sentences on every call using a generator. Does a lazy load of the data.

        If augment is False, then original sentences are returned until end of file is reached. Useful if corpus is large and you cannot load the whole data into memory.

        Else if augment is True, you can keep cycling through the dataset generating new augmented versions of the sentences on each cycle.
        """

        # Returning original sentences as they are if self.augment is False.
        if not self.augment:
            return next(self.src_input_file).rstrip('\n'), next(self.tgt_input_file).rstrip('\n')

        # Converting sample (string of sentences) to stanza.models.common.doc.Document object.
        src_sents = self.src_pipeline(next(self.src_input_file).rstrip('\n'))
        tgt_sents = self.tgt_pipeline(next(self.tgt_input_file).rstrip('\n'))

        # Placeholder list to hold augmented sentences, will join all sentences in sample before returning.
        augmented_src_sents = list()
        augmented_tgt_sents = list()

        # Using separate for loops (and not zipping) because source and target samples may have different number of sentences.
        for src_sent in src_sents.sentences:
            augmented_src_sents.append(depparse_aug(src_sent, self.mode, self.alpha, self.src_freq_dict))
        for tgt_sent in tgt_sents.sentences:
            augmented_tgt_sents.append(depparse_aug(tgt_sent, self.mode, self.alpha, self.tgt_freq_dict))

        # Joining all sentences.
        augmented_src_sents = ' '.join(augmented_src_sents)
        augmented_tgt_sents = ' '.join(augmented_tgt_sents)

        return augmented_src_sents, augmented_tgt_sents