#!/bin/bash
which nvidia-docker
if [ $? -eq 0 ]; then
	DOCKER_CMD=`/usr/bin/which nvidia-docker`
else
	DOCKER_CMD=`/usr/bin/which docker`
fi

DATA_DIR="../data_aqqu"
if [ ! -d $DATA_DIR ]; then
	DATA_DIR="../aqqu/data"
fi

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

VOLUME="$(pwd)/$DATA_DIR:/app/data"
if [ "$1" == "learner" ]; then
	echo "Learner"
	$DOCKER_CMD run --rm -it --name "aqqu_$1_$2_inst" -v $VOLUME  "aqqu_$1_$2"
elif [ "$1" == "backend" ]; then
	echo "Backend"
	$DOCKER_CMD run --rm -d -p 0.0.0.0:$PORT:8090 --name "aqqu_$1_$2_inst" -v $VOLUME "aqqu_$1_$2"
else
	echo "Debug"
	$DOCKER_CMD run --rm -it -p 0.0.0.0:$PORT:8090 --name "aqqu_$1_$2_inst" -v $VOLUME "aqqu_$1_$2"
fi
