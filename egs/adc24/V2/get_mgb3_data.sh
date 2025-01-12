#!/bin/bash
#
#           2024 Ecole de Technologie Superieure (Mohammed Akram Khelfi)
# Apache 2.0.
set -e



. ./path.sh
. datapath.sh
cmd="run.pl"

config_file=default_config.sh

echo "$0 $@"  # Print the command line for logging
. parse_options.sh || exit 1;
. $config_file
. datapath.sh





echo "Downloading MGB3 data for train"

$cmd data/mgb3/train/log/download.log \
        hyp_utils/conda_env.sh \
        local/get_mgb3.py \
        --wav-list data/mgb3/train/wav.lst \
        --split-id train \
        --files-output data/mgb3/train \
        --output-dir /export/b14/mkhelfi1/hyp-data/MGB3-ADI 

echo "Succeded downloading MGB3 data for train"


echo "Downloading MGB3 data for train"

$cmd data/mgb3/dev/log/download.log \
        hyp_utils/conda_env.sh \
        local/get_mgb3.py \
        --wav-list data/mgb3/dev/wav.lst \
        --split-id dev \
        --files-output data/mgb3/dev \
        --output-dir /export/b14/mkhelfi1/hyp-data/MGB3-ADI 

echo "Succeded downloading MGB3 data for train"
