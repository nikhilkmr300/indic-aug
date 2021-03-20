.SILENT:

vocab: make_vocab.py preprocessed ../.config
	python3 make_vocab.py

preprocessed: make_preprocessed.py raw ../.config raw/train.en
	python3 make_preprocessed.py

raw/train.en:
	cd raw && make

# Removes only vocab, keeping preprocessed data. Useful if you only need to retrain the vocab.
# Unless you've made changes to the preprocessing pipeline, running make mostlyclean should be sufficient.
mostlyclean:
	rm -rf vocab

# Removes vocab and preprocessed data. Running make again will perform preprocessing again.
clean:
	rm -rf vocab
	rm -rf preprocessed