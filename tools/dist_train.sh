#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

python -m torch.distributed.launch --nproc_per_node=$GPUS --master-port=$PORT \
    tools/train.py $CONFIG --launcher pytorch