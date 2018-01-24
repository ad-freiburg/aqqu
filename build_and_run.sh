#!/bin/bash
which nvidia-docker
if [ $? -eq 0 ]; then
	DOCKER_CMD=`/usr/bin/which nvidia-docker`
else
	DOCKER_CMD=`/usr/bin/which docker`
fi

INPUT_DIR="./input"
WORKDATA_DIR="./data"

if [ $# -lt 2 ] || [ "$1" != "backend" ] && [ "$1" != "learner" ] && [ "$1" != "debug" ]; then
	echo "Usage: $0 [backend|debug|learner] name [PORT]"
	exit 1
fi

PORT=8090
if [ $# -gt 2 ];then
	PORT=$3
fi
echo "-----------------------------------------------------------------"
echo Executing $DOCKER_CMD build -t "aqqu_$1_$2" -f "Dockerfile.$1" .
$DOCKER_CMD build -t "aqqu_$1_$2" --build-arg LEARNER_BASE=$2 -f "Dockerfile.$1" .
echo "-----------------------------------------------------------------"

INPUT_VOLUME="$(pwd)/$INPUT_DIR:/app/input"
WORKDATA_VOLUME="$(pwd)/$WORKDATA_DIR:/app/data"
MODELS_VOLUME="aqqu_learner_$2_models_vol:/app/models"
if [ "$1" == "learner" ]; then
	echo "Learner"
	$DOCKER_CMD run --rm -it --name "aqqu_$1_$2_inst" \
		-v $INPUT_VOLUME \
		-v $WORKDATA_VOLUME \
		-v $MODELS_VOLUME \
		"aqqu_$1_$2"
elif [ "$1" == "backend" ]; then
	echo "Backend"
	$DOCKER_CMD run --rm -d -p 0.0.0.0:$PORT:8090 --name "aqqu_$1_$2_inst" \
		-v $INPUT_VOLUME  \
		-v $WORKDATA_VOLUME \
		-v $MODELS_VOLUME \
		"aqqu_$1_$2"
else
	echo "Debug"
	$DOCKER_CMD run --rm -it -p 0.0.0.0:$PORT:8090 --name "aqqu_$1_$2_inst" \
		-v $INPUT_VOLUME  \
		-v $WORKDATA_VOLUME \
		-v $MODELS_VOLUME \
		"aqqu_$1_$2"
fi
