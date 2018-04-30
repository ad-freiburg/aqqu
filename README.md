# Aqqu Question Answering System

This is the code accompanying the publication "More Accurate Question Answering on Freebase, Hannah Bast and Elmar Haussmann, CIKM 2015"

Follow the instructions in QUICKSTART.md to set up the system. This also includes descriptions on how to 
obtain pre-requisite data, e.g., a complete index of Freebase.

Setup is easy if all pre-requisites are met.

## Requirements:

* OS: Linux system (tested on Ubuntu 12.04)
* Software: Python 2.7 as well as Java 8 + Prerequisites for Virtuoso, see
  QUICKSTART.md for more details.
* RAM: 40 GB for training the large WebQuestions models
* Disk: about 40 GB for all pre-requisite data

## Get the Dataset

All data required for learning can be found under
`/nfs/datastets/aqqu_input_data`, all other data is generated automatically.

    cp -r /nfs/datasets/aqqu_input_data/* input/

## Train with the provided script
When using docker/wharfer with user namespaces you may need to first run
`chmod o+wx data` so Aqqu running inside docker can write data even if it is
a very restricted user like `nobody`

    ./build_and_run.sh train -n <user_provided_name> -r <ranker e.g. WQSP_Ranker> <additional args>

## Run with the provided script

    ./build_and_run.sh backend -n <user_provided_name> -r <ranker e.g. WQSP_Ranker> -p <port> <addtional args>

## Debug changes with the provided script
The `backend` command does not rebuild the image but reuses the exact image
used for training. Therefore changes to the source code made after the train
step are not reflected in the behavior of `backend`.

To try out changes **without affecting the train/backend** image use the debug
command.

    ./build_and_run.sh debug -n <user_provided_name> -r <ranker e.g. WQSP_Ranker> -p <port> <addtional args>

In the future I will probably add a command to create a new version of the
training image without rerunning the training. Note however that this is
somewhat dangerous as it does not work for changes that would make the model
incompatible to the trained model.

## Run Cross Validation with the provided script

    ./build_and_run.sh cv -n <user_provided_name> -r <ranker> <dataset name>

## Overriding parameters
To override certain ranker parameters you can use `--override` with a JSON object as additional argument for example
`--override '{"top_ngram_percentile": 15}'`

## Disabling GPU
To disable GPU use run above commands with the environment variable `NO_GPU=1`

## Using (nvidia-)docker directly
To use (nvidia-)docker directly refer to the `build_and_run.sh` script. No
additional documentation is provided as this is discouraged and keeping the
instructions up to date for a very rare use case is not worth the effort since
the script is easy to understand anyway.
