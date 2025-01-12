#!/bin/bash
# Copyright       2018   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh
nnet_stage=1

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

if [ $nnet_stage -eq 1 ];then
  nnet=$nnet_s1
  nnet_name=$nnet_s1_name
elif [ $nnet_stage -eq 2 ];then
  nnet=$nnet_s2
  nnet_name=$nnet_s2_name
elif [ $nnet_stage -eq 3 ];then
  nnet=$nnet_s3
  nnet_name=$nnet_s3_name
fi

epoch=7
transducer_text_dir=/export/fs05/mkhelfi1/hyperion/qasr/exp/transducer/wav2vec2xlsr300m_qasr_transducer_500_4gpu_v1.0.s1/data/qasr/dev_proc_audio/transducer.text
output_dir=exp/scores/$nnet_name/$epoch/log
if [ $stage -le 1 ]; then
  mkdir -p $output_dir
  $train_cmd $output_dir/eval_wer.log \
  hyp_utils/conda_env.sh --conda-env $HYP_ENV \
  local/eval_wer.py \
  --source-text $dev_data/text \
  --target-text $transducer_text_dir \

fi

