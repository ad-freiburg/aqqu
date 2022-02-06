# Aqqu Question Answering System

This is an ancestor of the code accompanying the publication: [More Accurate
Question Answering on Freebase, Hannah Bast and Elmar Haussmann, CIKM 2015](https://ad-publications.cs.uni-freiburg.de/freebase-qa.pdf)

The original version can be found as as the very first commit of this
repository and under the v1.0 tag.

Follow the instructions below to set up the system. This also includes
descriptions on how to obtain pre-requisite data.

Setup is easy if all pre-requisites are met.

## Requirements:

* Docker > 18.09 on 64 bit Linux
* RAM: 40 GB for training the large WebQuestions/WebQSP models
* Disk: about 50 GB for all required data

## Setup a Virtuoso Instance with Freebase

To setup a Virtuoso instance with the Freebase data needed for Aqqu we
recommend to follow the instructions at our  [Virtuoso with Docker
Compose](https://github.com/ad-freiburg/virtuoso-compose) repository. This will
automatically download the correct version of Freebase and setup Virtuoso with
the exact same settings used by us.

## Get the Dataset

All data required for learning can be found under
`/nfs/datastets/aqqu_input_data` when on any of the Chair's computer systems,
all other data is generated automatically.

    cp -r /nfs/datasets/aqqu_input_data/* input/

Outside our system's please contact [Prof. Hannah
Bast](http://ad.informatik.uni-freiburg.de/contact). with the above path to get
a download.

## Train with the provided script
When using docker/wharfer with user namespaces you may need to first run
`chmod o+wX data` so Aqqu running inside docker can write data even if it is
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


Once the changes have been tested the image used by the `backend` command can
be updated using. *Note* however that this is somewhat dangerous as it does not
work for changes that would make the model incompatible to the trained model.

    ./build_and_run.sh update -n <user_provided_name>

Test the Backend
----------------
The Aqqu backend provides a simple JSON API that can easily be tested using
`curl`. If you have the `./build_and_run.sh backend …` command running on
a server `<host>` with port `<port>` the following asks for Albert Einstein's
place of birth

    curl http://<host>:<port>/?q=where%20was%20albert%einstein%20born

## Run Cross Validation with the provided script

    ./build_and_run.sh cv -n <user_provided_name> -r <ranker> <dataset name>

## Overriding parameters
To override certain ranker parameters you can use `--override` with a JSON object as additional argument for example
`--override '{"top_ngram_percentile": 15}'`

## Disabling GPU
To disable GPU use run above commands with the environment variable `NO_GPU=1`

## Using (nvidia-)docker directly
To use (nvidia-)docker directly refer to the `build_and_run.sh` script. No
additional documentation is provided as this is discouraged the script should
be reasonably easy to follow.
