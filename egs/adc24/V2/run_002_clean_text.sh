#!/bin/bash
# Copyright
#                2024   Johns Hopkins University (Author: Mohammed Akram Khelfi)
#
# 

. ./cmd.sh
. ./path.sh

config_file=default_config.sh
stage=1


. parse_options.sh || exit 1;
. ./datapath.sh 
. $config_file


# Qasr Text Cleaning 

# if [ $stage -le 1 ];then
#    echo " Stage 1: Qasr Text Cleaning"
#    for name in dev test train; do
#     echo " Processing Text for $name"
#     mkdir -p data/qasr/${name}_proc_audio/log
#       $cuda_cmd \
#       --gpu 0 data/qasr/${name}_proc_audio/log/clean_text.log \
#     hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus 0 \
#     local/clean_text.py \
#     --source-file data/qasr/${name}_proc_audio/text \
#     --target-file data/qasr/${name}_proc_audio/text_clean
    
#    done

# fi

if [ $stage -le 2 ];then
  echo "Stage 2: Clean LM text"
  mkdir -p data/qasr/lm/log
  $cuda_cmd \
  --gpu 0 data/qasr/lm/log/clean_text.log \
  hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus 0 \
  local/clean_text.py \
  --source-file data/qasr/lm/qasr_transcript_words.txt \
  --target-file data/qasr/lm/qasr_transcript_words_clean.txt \
  --text-only
fi