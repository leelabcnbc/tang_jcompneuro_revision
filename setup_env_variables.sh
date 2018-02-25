#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

TANG_PATH="${DIR}"
LEELAB_BOX_PATH="${HOME}/leelab-toolbox"
STRFLAB_PATH="${HOME}/strflab-python"
export PYTHONPATH="${TANG_PATH}":"${LEELAB_BOX_PATH}":"${STRFLAB_PATH}":"${PYTHONPATH}"
