.SILENT:

.PHONY: all mostlyclean clean

# Generates concatenated train, dev and test sets.
train.en: concat.sh finalrepo
	./concat.sh

# Extracts tar file.
finalrepo: indic_wat_2021.tar
	echo "Extracting indic_wat_2021.tar..."
	# Using --modification-time as tar by defualt extracts time when archived (so make reruns unnecessarily).
	tar --modification-time -xf indic_wat_2021.tar

# Removes generated train, dev and test sets, but retains unzipped tar.
mostlyclean:
	rm -f *.en
	rm -f *.hi

# Removes all generated files.
clean:
	rm -rf finalrepo
	rm -f *.en
	rm -f *.hi