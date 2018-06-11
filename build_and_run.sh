#!/bin/bash
TENSORFLOW=gcr.io/tensorflow/tensorflow:latest-py3

which nvidia-docker
if [ $? -eq 0 ] && ! [ -v NO_GPU ]; then
	DOCKER_CMD=`/usr/bin/which nvidia-docker`
	TENSORFLOW=gcr.io/tensorflow/tensorflow:latest-gpu-py3
else
	which wharfer
	if [ $? -eq 0 ]; then
		DOCKER_CMD=`/usr/bin/which wharfer`
	else
		DOCKER_CMD=`/usr/bin/which docker`
	fi
fi

function help {
	echo "Usage: $0 train|cv|backend|debug|update [--port PORT] [--name name] [--no-cache] [--ranker ranker] [ARGS..]"
	exit 1
}

INPUT_DIR="./input"
WORKDATA_DIR="./data"
PORT=8090
RANKER="WQSP_Ranker"
NAME="wqsp_default"
CACHE=""

POSITIONAL=()
while [[ $# -gt 0 ]]
do
	key="$1"

	case $key in
			-p|--port)
				PORT="$2"
				shift # shift argument
				shift # shift value
			;;
			-h|--help)
				help # exits
			;;
			-nc|--no-cache)
				CACHE="--pull --no-cache"
				shift # only have an argument
			;;
			-n|--name)
				NAME="$2"
				shift # shift argument
				shift # shift value
			;;
			-r|--ranker)
				RANKER="$2"
				shift # shift argument
				shift # shift value
			;;
			*)    # unknown option
				POSITIONAL+=("$1") # save it in an array for later
				shift # shift argument
			;;
	esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

if [ "$1" != "backend" ] && [ "$1" != "train" ] && \
	 [ "$1" != "debug" ] && [ "$1" != "update" ] && [ "$1" != "cv" ]; then
	help
fi


INPUT_VOLUME="$(pwd)/${INPUT_DIR}:/app/input"
WORKDATA_VOLUME="$(pwd)/${WORKDATA_DIR}:/app/data"
chmod o+wx ${WORKDATA_DIR}

MODELS_DIR="aqqu_learner_${NAME}_models"
# Create a model directory if it doesn't exist
mkdir -p ${MODELS_DIR}
chmod o+wx ${MODELS_DIR}
# Add .gitignore so the models don't end up in git, '>' overwrites if it exists but that's ok
echo '!.gitignore' > ${MODELS_DIR}/.gitignore
MODELS_VOLUME="$(pwd)/${MODELS_DIR}:/app/models"

function build {
	echo "-----------------------------------------------------------------"
	echo Executing $DOCKER_CMD build
	$DOCKER_CMD build ${CACHE} -t "aqqu_$1" \
		--build-arg TENSORFLOW=$TENSORFLOW \
		-f "Dockerfile" .
}

if [ "$1" == "train" ] || [ "$1" == "cv" ]; then
	echo "Learner"
	build ${NAME}
	echo "-----------------------------------------------------------------"
	$DOCKER_CMD run --rm -it --init --name "aqqu_$1_${NAME}_inst" \
		-v $INPUT_VOLUME \
		-v $WORKDATA_VOLUME \
		-v $MODELS_VOLUME \
		"aqqu_${NAME}" query_translator.learner "$1" "${RANKER}" "${@:2}"
elif [ "$1" == "backend" ]; then
	echo "Backend"
	$DOCKER_CMD run --restart unless-stopped --init -d -p ${PORT}:8090 \
		--name "aqqu_$1_${NAME}_inst" \
		-v $INPUT_VOLUME  \
		-v $WORKDATA_VOLUME \
		-v $MODELS_VOLUME \
		"aqqu_${NAME}" translator_server "${RANKER}" "${@:2}"
elif [ "$1" == "debug" ]; then
	echo "Debug"
	build debug_${NAME}
	$DOCKER_CMD run --rm -it -p $PORT:8090 --init --name "aqqu_$1_${NAME}_inst" \
		-v $INPUT_VOLUME  \
		-v $WORKDATA_VOLUME \
		-v $MODELS_VOLUME \
		"aqqu_debug_${NAME}" translator_server "${RANKER}" "${@:2}"
elif [ "$1" == "update" ]; then
	echo "-----------------------------------------------------------------"
	echo "Update"
	build ${NAME}
fi
