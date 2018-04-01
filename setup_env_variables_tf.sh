#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# setup cudnn and cuda
. ~/DevOps/env_scripts/add_cuda_lib_v9.sh
. ~/DevOps/env_scripts/add_cudnn_v7.sh

KERAS_VIS_PATH="${HOME}/keras-vis"

export PYTHONPATH="${KERAS_VIS_PATH}":"${PYTHONPATH}"
