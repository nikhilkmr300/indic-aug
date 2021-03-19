.SILENT:

vocab: make_vocab.py preprocessed ../.config
	python3 make_vocab.py
	# Do not need raw once we have vocab.
	cd raw && make mostlyclean

preprocessed: make_preprocessed.py ../.config
	cd raw && make
	python3 make_preprocessed.py

clean:
	rm -rf vocab
	rm -rf preprocessed
	cd raw && make mostlyclean