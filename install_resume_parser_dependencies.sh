#!/bin/bash

# Install pyresparser
pip install pyresparser

# Install en_core_web_sm spaCy model
python -m spacy download en_core_web_sm

# Download words and stopwords for nltk
python -m nltk.downloader words
python -m nltk.downloader stopwords
python -m nltk.downloader punkt
python -m nltk.downloader wordnet

pip install -r requirements_resume_parsing.txt