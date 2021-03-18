.SILENT:

vocab: make_vocab.py preprocessed
	python3 make_vocab.py
	# Do not need preprocessed once we have vocab.
	rm -rf preprocessed

preprocessed: make_preprocessed.py
	cd raw && make
	python3 make_preprocessed.py

clean:
	rm -rf vocab
	rm -rf preprocessed
	cd raw && make mostlyclean