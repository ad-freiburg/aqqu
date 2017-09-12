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
	echo "Usage: $0 [backend|learner] name [PORT]"
	exit 1
fi

PORT=8090
if [ $# -gt 2 ];then
	PORT=$3
fi
echo "-----------------------------------------------------------------"
echo Executing $DOCKER_CMD build -t "aqqu_$1_$2" -f "Dockerfile.$1" .
$DOCKER_CMD build -t "aqqu_$1_$2" -f "Dockerfile.$1" .
echo "-----------------------------------------------------------------"
if [ "$1" == "learner" ]; then
	echo Executing $DOCKER_CMD run --rm -it --name "aqqu_$1_$2_inst" -v "$(pwd)/$DATA_DIR:/app/data" "aqqu_$1_$2"
	$DOCKER_CMD run --rm -it --name "aqqu_$1_$2_inst" -v "$(pwd)/$DATA_DIR:/app/data" "aqqu_$1_$2"
else
	echo Executing $DOCKER_CMD run --rm -p 0.0.0.0:$PORT:8090 -d --name "aqqu_$1_$2_inst" -v "$(pwd)/$DATA_DIR:/app/data" "aqqu_$1_$2"
	$DOCKER_CMD run --rm -d -p 0.0.0.0:$PORT:8090 --name "aqqu_$1_$2_inst" -v "$(pwd)/$DATA_DIR:/app/data" "aqqu_$1_$2"
fi
