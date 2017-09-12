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

if [ $# -lt 2 ] || [ "$1" != "backend" ] && [ "$1" != "learner" ]; then
	echo "Usage: $0 [backend|learner] name "
	exit 1
fi

$DOCKER_CMD build -t "aqqu_$1_$2" -f "Dockerfile.$1" .

if [ "$1" == "learner" ]; then
	$DOCKER_CMD run --rm -it --name "aqqu_$1_$2_inst" -v "$(pwd)/$DATA_DIR:/app/data" "aqqu_$1_$2"
else
	$DOCKER_CMD run --rm -d --name "aqqu_$1_$2_inst" -v "$(pwd)/$DATA_DIR:/app/data" "aqqu_$1_$2"
fi
