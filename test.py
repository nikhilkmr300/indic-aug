# # Dependency parsing demo
# # -----------------------
# import stanza
# import numpy as np

# from indic_aug.depparse.depparse import DepParseTree, depparse_aug

# np.random.seed(10)

# pipeline = stanza.Pipeline(lang='en')
# sent = 'Jack hit the ball with the bat'
# sent = pipeline(sent).sentences[0]
# tree = DepParseTree(sent)
# tree.save_tree('output.gv')
# print(depparse_aug(sent, 'blank', 0.3))

# # Noising demo
# # ------------
# from indic_aug.basic import NoisingAugmentor
# from tqdm import tqdm
# aug = NoisingAugmentor('example/data/preprocessed/dev.en', 'example/data/preprocessed/dev.hi', 'replace', 0.1)
# for i in tqdm(range(len(aug))):
#     print(next(aug))

# # Dropout demo
# # ------------
# from indic_aug.basic import DropoutAugmentor
# from tqdm import tqdm
# aug = DropoutAugmentor('example/data/preprocessed/dev.en', 'example/data/preprocessed/dev.hi', 0.3)
# for i in tqdm(range(len(aug))):
#     print(next(aug))

# # Synonym demo
# # ------------
# from indic_aug.basic import SynonymAugmentor
# aug = SynonymAugmentor('example/data/preprocessed/dev.en', 'example/data/preprocessed/dev.hi', 0.5, random_state=1)
# print(next(aug))

# # Alignment demo
# # --------------
# import dill as pickle
# import time

# from indic_aug.align import Aligner

# ibm1 = Aligner('ibm1', 5)

# # # Training model.
# # t1 = time.time()
# # ibm1.train('example/data/preprocessed/train.en', 'example/data/preprocessed/train.hi')
# # t2 = time.time()
# # print(f'Training IBM Model 1 took {t2 - t1} s.')
# # ibm1.serialize('ibm1.pkl')

# # Loading model.
# with open('ibm1.pkl', 'rb') as f:
#     ibm1 = pickle.load(f)

# test_words = [
#     'John',
#     'hit',
#     'the',
#     'ball',
#     'with',
#     'the',
#     'bat',
#     '.'
# ]

# print('IBM Model 1\n' + '-' * len('IBM Model 1'))
# for test_word in test_words:
#     print('{:20s}{:20s}'.format(test_word, ibm1.get_aligned_word(test_word)))

# Top n and bottom n words
# ------------------------
import os

from indic_aug.vocab import read_topn_vocab, read_bottomn_vocab

model_path = os.path.join('example', 'data', 'vocab', 'en.model')
vocab_path = os.path.join('example', 'data', 'vocab', 'en.vocab')
n = 100

print(read_topn_vocab(model_path, vocab_path, n).keys())
print(read_bottomn_vocab(model_path, vocab_path, n).keys())