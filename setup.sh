#! /bin/bash

# Script to setup the conda environment with the required packages.

conda init bash
conda activate nmt

if [ $? -eq 1 ]; then
    echo -e "Environment 'nmt' nor found. Creating environment 'nmt' and installing packages...\n"
    # If environment does not exist, creating it.
    echo "y" | conda create --name nmt python=3.7.0
    conda init bash
    conda activate nmt
fi

conda_packages=(
    "pip"
    "numpy"
    "scipy"
    "pandas"
    "tqdm"
    "nltk"
    "dill"
    "sacremoses"
    "sentencepiece"
    "graphviz"
    "python-graphviz"
    "sphinx"
    "sphinxcontrib-bibtex"
    "sphinx_rtd_theme"
)
conda config --add channels conda-forge

pip_packages=(
    "stanza"
    "pyiwn"
    "polyglot"
    "indic-nlp-library"
)

# Installing conda packages.
for conda_package in ${conda_packages[@]}; do
    conda list | grep "$conda_package" > /dev/null
    if [[ $? -eq 0 ]]; then
        printf "%s already installed by conda.\n" "$conda_package"
        continue
    fi
    echo "y" | conda install $conda_package
done

# Installing pip packages not available in conda.
for pip_package in ${pip_packages[@]}; do
    pip list | grep "$pip_package" > /dev/null
    if [[ $? -eq 0 ]]; then
        printf "%s already installed by pip.\n" "$pip_package"
        continue
    fi
    pip install $pip_package
done

# Removing torch>1.3.0 installed as dependency of stanza.
echo "y" | pip uninstall torch
# Installing torch==1.3.0 as required by inltk.
echo "y" | conda install pytorch==1.3.0 -c pytorch