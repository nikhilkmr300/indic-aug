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
# from indic_aug.align import Aligner
# aligner = Aligner('ibm1', 5)
# print('Training...')
# aligner.train('example/data/preprocessed/dev.en', 'example/data/preprocessed/dev.hi')
# aligner.serialize('sample.pkl')