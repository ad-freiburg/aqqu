#!/bin/bash
TENSORFLOW=gcr.io/tensorflow/tensorflow:latest-py3

which nvidia-docker
if [ $? -eq 0 ] && ! [ -v NO_GPU ]; then
	DOCKER_CMD=`/usr/bin/which nvidia-docker`
	TENSORFLOW=gcr.io/tensorflow/tensorflow:latest-gpu-py3
else
	DOCKER_CMD=`/usr/bin/which docker`
fi

INPUT_DIR="./input"
WORKDATA_DIR="./data"

if [ $# -lt 2 ] || [ "$1" != "backend" ] && [ "$1" != "learner" ] && [ "$1" != "debug" ]; then
	echo "Usage: $0 [learner|backend|debug] name ranker [PORT]"
	exit 1
fi

PORT=8090
if [ $# -gt 3 ];then
	PORT=$4
fi

INPUT_VOLUME="$(pwd)/$INPUT_DIR:/app/input"
WORKDATA_VOLUME="$(pwd)/$WORKDATA_DIR:/app/data"
MODELS_VOLUME="aqqu_learner_$2_models_vol:/app/models"
if [ "$1" == "learner" ]; then
	echo "Learner"
	echo "-----------------------------------------------------------------"
	echo Executing $DOCKER_CMD build
	$DOCKER_CMD build -t "aqqu_$2" \
		--build-arg TENSORFLOW=$TENSORFLOW \
		-f "Dockerfile.base" .
	echo "-----------------------------------------------------------------"
	$DOCKER_CMD run --rm -it --name "aqqu_$1_$2_inst" \
		-v $INPUT_VOLUME \
		-v $WORKDATA_VOLUME \
		-v $MODELS_VOLUME \
		"aqqu_$2" query_translator.learner train $3
elif [ "$1" == "backend" ]; then
	echo "Backend"
	$DOCKER_CMD run --rm -d -p 0.0.0.0:$PORT:8090 --name "aqqu_$1_$2_inst" \
		-v $INPUT_VOLUME  \
		-v $WORKDATA_VOLUME \
		-v $MODELS_VOLUME \
		"aqqu_$2" translator_server $3
else
	echo "Debug"
	echo Executing $DOCKER_CMD build
	$DOCKER_CMD build -t "aqqu_debug_$2" \
		--build-arg TENSORFLOW=$TENSORFLOW \
		-f "Dockerfile.base" .
	$DOCKER_CMD run --rm -it -p 0.0.0.0:$PORT:8090 --name "aqqu_$1_$2_inst" \
		-v $INPUT_VOLUME  \
		-v $WORKDATA_VOLUME \
		-v $MODELS_VOLUME \
		"aqqu_debug_$2" translator_server $3
fi
