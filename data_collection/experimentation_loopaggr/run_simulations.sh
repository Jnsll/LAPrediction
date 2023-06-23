#!/bin/sh

#Parameters for the computation on the server
CPU=1
CPU_EXP=1
MEMORY='2g'
PATH_DOCKER_SOCK='/var/run/docker.sock'
PWD_DIR=$(pwd)
DIR=$(dirname "$PWD_DIR")/results 
ID_USER=$(id -u $USER)
DATE=$(date +'%H:%M_%d-%m-%Y')
# Parameters to run the simulation
FOLDER='' #To complete : the path of the directory where the results will be stored
SITE=$1   # Int: Id of the geographical site (i.e., watershed)
CHR=$2    # Int: Id of the climate scenario
APPROX=$3 # Int: Id of the fonction for the iteration aggregation
RATE=$4   # Int: value of the upscaling factor
REF=$5    # Boolean: if the simulation is the reference one (no discretization upscale)
PERM=$6   # Float: value of the permeability parameter
STEADY=$7 # Boolean: wether the water flows are considered steady
REP=$8    #Int: Id of the simulation replication
  

if [ $CPU -ge $CPU_EXP ]
then

  PATH_DOCKER_SOCK=$PATH_DOCKER_SOCK':/var/run/docker.sock'

  echo "--------------------------------------------"
  echo 'cpu allowed to each simulation: '$CPU_EXP
  echo 'memory allowed to each simulation: '$MEMORY
  echo "--------------------------------------------------------"
  echo " (1 / 4) creation of the directory entitled results/ ..."
  echo " -------------------------------------------------------"
  mkdir -p ../results;

  echo "-------------------------------------------"
  echo " (2 / 4) build modflow-simulation-docker..."
  echo " ------------------------------------------"
  docker build -t modflow-simulation-docker -f docker-simulation/Dockerfile ./docker-simulation;

  echo "-----------------------------------"
  echo " (3 / 4) run modflow-main-docker..."
  echo "-----------------------------------"

  if [ $REF == '1' ]
  then
    REF="-ref"
  else
    REF=""
  fi
  echo $FOLDER
  docker run -e LANG=C.UTF-8 --rm -v $FOLDER:/modflow/outputs --memory $MEMORY --cpus=1 -e SITE=$SITE -e APPROX=$APPROX -e RATE=$RATE -e CHR=$CHR -e REF=$REF -e REP=$REP -e STEADY=$STEADY -e PERM=$PERM modflow-simulation-docker;  
  echo "-----------------------------------------------------------"
  echo " (4 / 4) simulations over, check results/ to see the data.."
  echo "-----------------------------------------------------------"
else
  echo "--> Error: cpu allowed to each simulation higher than cpu allowed to all the program"
fi

