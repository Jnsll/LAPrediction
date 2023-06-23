#!/bin/sh

#Parameters for the computation on the server
CPU=20
CPU_EXP=5
MEMORY='15g'
PATH_DOCKER_SOCK='/var/run/docker.sock'
PWD_DIR=$(pwd)
DIR=$(dirname "$PWD_DIR")/results
ID_USER=$(id -u $USER)
DATE=$(date +'%H:%M_%d-%m-%Y')
FOLDER=''
SITE=$1
CHR=$2
APPROX=$3
RATE=$4
REF=$5
PERM=$6
STEADY=$7
TOPO=$8

echo "DIR variable"
echo $DIR
echo $DIR/



if [ $CPU -ge $CPU_EXP ]
then

  PATH_DOCKER_SOCK=$PATH_DOCKER_SOCK':/var/run/docker.sock'

  echo "-------------------------------------------"
  echo " (1 / 4) build modflow-simulation-docker..."
  echo " ------------------------------------------"
  docker build -t modflow-simulation-docker-cm -f docker-simulation/Dockerfile.Cm ./docker-simulation;


  echo "-----------------------------------------"
  echo " (2 / 4) run modflow-simulation-docker..."
  echo "-----------------------------------------"
  docker run --rm -v $FOLDER:/modflow/outputs --cpus=1 -e LANG=C.UTF-8 -e RATE=$RATE -e APPROX=$APPROX -e CHR=$CHR -e SITE=$SITE -e REF=$REF -e PERM=$PERM -e STEADY=$STEADY -e TOPO=$TOPO modflow-simulation-docker-cm;

  echo "-----------------------------------------------------------"
  echo " (4 / 4) simulations over, check results/ to see the data.."
  echo "-----------------------------------------------------------"
else
  echo "--> Error: cpu allowed to each Cm computation is higher than cpu allowed to all the program"
fi
