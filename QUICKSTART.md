# Installing Aqqu

This document describes how to set up, run and evaluate the Aqqu QA system.

## Installing

### Requirements

Running the code initially requires around 25 GB of free RAM.
After the inital start and some pre-computation around 10-15 GB are enough.

The code has been tested on Ubuntu 12.04 but should work on any recent
Linux distribution with Python >= 2.7.

### Required Packages

Pre-requisite Python packages are listed in the file requirements.txt. These are
mainly scikit-learn (for machine learning), which requires numpy and scipy,
Cython (for generation of fast C extensions) as well as a few smaller libraries.
If you have virtualenv and pip, setup is easy:

    virtualenv -p python3 venv
    source venv/bin/activate
    pip install -r requirements.txt

This creates a virtual environment containing a copy of your Python interpreter.
All packages will be installed into that environment. Installation takes around
45 minutes. If anything goes wrong, check which package installation failed, and
refer to the documentation of the corresponding package.

After the installation you need to compile some low-level code:

    python setup.py build_ext --inplace

### Install Virtuoso Opensource

A Virtuoso V7 instance that provides an index on the Freebase triples is
required. We provide the index for download. To install Virtuoso run:

    make install-virtuoso

This will check out the source, build it and install into the subdirectory
"virtuoso-opensource" (which takes approximately 30 minutes). You need to have all
requirements installed (bison, flex, etc.). Refer to the installation
[documentation](https://github.com/openlink/virtuoso-opensource) of Virtuoso for details.
Alternatively, you can install it yourself or use an existing installation.
Then you need to update the path to the Virtuoso binary in the Makefile (VIRTUOSO_BINARY).

### Download Pre-requisite Data

All pre-computed data (computed indicator words, word embeddings etc.) are downloaded
into the data directory "data". This is approximately 20 GB of data,
so make sure you have enough space. To download the data run:

    make download-data

This downloads the (compressed) data and extracts it. It also downloads and
extracts the virtuoso index into the "virtuoso-db" and the Stanford CoreNLP library
(for POS-tagging) into the "lib" directory.

### Configuration

Most configuration is part of a single config file: config.cfg
You need to update the base parameter in the DEFAULT section to point to the
directory the config file resides, e.g.: /home/username/aqqu .Besides that, the
default settings should work. In case you choose to use a different SPARQL
backend or parser make sure to update the values in the section SPARQLBackend
and CoreNLPParser.

## Running

### Start SPARQL and CoreNLP Components

You first need to start the SPARQL and CoreNLP components:

    make start-virtuoso
    make start-parser

Check the output to confirm both services are running. The parser uses Stanford
CoreNLP which requires Java 8 and Ant.

### Train and evaluate models

The learner module can be used to train and test models. To execute it run:

    python -m query_translator.learner

It will output its parameters and options. The module will also cache executed
queries and candidates which allows faster training and testing of rankers
in subsequent runs.

#### Models and Datasets

Models and datasets are defined in the scorer_globals.py file. Rankers are
implemented in the query_translator.ranker module. Each model/ranker has a
unique name and a fixed training dataset.

The important models/rankers are:

 - F917_Ranker: ranker for Free917 dataset
 - F917_Ranker_entity_oracle: ranker for Free917 dataset using an
   entity lexicon
 - WQ_Ranker: ranker for WebQuestions dataset

Important datasets (with obvious interpretation) are:

 - free917train
 - free917test
 - webquestionstrain
 - webquestionstest

The remaining datasets refer to splits or subsets of the above.

#### Reproduce Results

You first need to train each model. Run:

    python -m query_translator.learner train <ranker>

to train <ranker> (i.e. replace <ranker> with F917_Ranker_entity_oracle etc.).
This will also perform some one-time pre-processing and cache the translated queries
to disk, so that subsequent calls are a lot faster.

To reproduce the results run:

    python -m query_translator.learner test <ranker> <dataset>

I.e. replace <ranker> and <dataset> with, e.g., WQ_Ranker and webquestionstest.

When evaluating, statistics will be output at the very end. This will
also write a file "eval_out.log" containing questions, gold answers, and
predicted answers. You can use this file with the published evaluation script by
Berant et al. (evaluate.py) to compute the published results.

### Start console based interface

We also provide a (very simple) console based interface to translate questions.
Make sure pre-requisites have been installed. Don't forget to activate the
virtual environment if you have one (source venv/bin/activate). Then run:

    python console_translator.py <ranker>

where <ranker> is one of the above listed rankers. This will perform some one time
pre-processing and start a console based interface that will allow you to input a
question and provide you with the top-ranked answer.
