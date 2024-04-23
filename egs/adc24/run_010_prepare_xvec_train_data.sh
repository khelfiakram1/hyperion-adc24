#!/bin/bash
# Copyright
#                2024   Johns Hopkins University (Author: Mohammed Akram Khelfi)
#                2020   Johns Hopkins University (Author: Jesus Villalba)
# Apache 2.0.
#

. ./cmd.sh
. ./path.sh
set -e

stage=1
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file

if [ $stage -le 1 ]; then
  # This script preprocess audio for x-vector training
  for name in adi17_codecs
  do
    steps_xvec/preprocess_audios_for_nnet_train.sh \
      --nj 40 --cmd "$train_cmd" \
      --storage_name lre22-fixed-v1.8k-$(date +'%m_%d_%H_%M') --use-bin-vad true \
      data/${name} data/${name}_proc_audio_no_sil exp/${name}_proc_audio_no_sil
    utils/fix_data_dir.sh data/${name}_proc_audio_no_sil
  done
fi

