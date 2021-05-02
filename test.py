from indic_aug.preprocess import Preprocessor

raw_src_path = 'example/raw/data.hi'
raw_tgt_path = 'example/raw/data.en'
prevocab_src_path = 'pre.hi'
prevocab_tgt_path = 'pre.en'
out_src_path = 'post.hi'
out_tgt_path = 'post.en'

tokenizer = Preprocessor(raw_src_path, raw_tgt_path)
tokenizer.pre_vocab(prevocab_src_path, prevocab_tgt_path, 100)
tokenizer.build_vocab(2000, 2000, 'vocab_dir')
tokenizer.post_vocab(out_src_path, out_tgt_path)