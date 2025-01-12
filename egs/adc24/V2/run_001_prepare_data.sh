/#!/bin/bash
# Copyright
#                2024   Johns Hopkins University (Author: Mohammed Akram Khelfi)
#
# 

. ./cmd.sh
. ./path.sh

config_file=default_config.sh
stage=1

. parse_options.sh || exit 1;

. datapath.sh

audiodst="/export/fs05/mkhelfi1/QASR"
# #Qasr data preparation 
# if [ $stage -le 1 ];then
#     echo " Stage 1: Qasr Data Preparation"
#     local/qasr_data_prep.sh $qasr_root data/qasr $audiodst

# fi
if [ $stage -le 2 ];then
   echo " Stage 2: Data Conversion"
   for part in dev test train
   do
      echo ${part}
      # steps_transducer/preprocess_audios_for_nnet_train.sh --nj 20 --cmd "$train_cmd" \
      # --storage_name qasr-seg-$(date +'%m_%d_%H_%M') --use-bin-vad false \
      # --osr 16000 data/qasr/${part} data/qasr/${part}_proc_audio /export/fs05/mkhelfi1/hyp-data/qasr/${part}_proc_audio 
      #utils/fix_data_dir.sh data/qasr/${part}_proc_audio || true
      awk '{ 
         speaker_id = $2; 
         for (i=3; i<=NF; i++) speaker_id = speaker_id "_" $i; 
         print $1, speaker_id 
}'    data/qasr/${part}_proc_audio/utt2spk > data/qasr/${part}_proc_audio/fixed_utt2spk
   done
fi
# mkdir -p data/qasr/lm
# cut -d' ' -f1 --complement  data/qasr/train_proc_audio/text > data/qasr/lm/qasr_transcript_words.txt

awk '{ 
    speaker_id = $2; 
    for (i=3; i<=NF; i++) speaker_id = speaker_id "_" $i; 
    print $1, speaker_id 
}' data/qasr/dev_proc_audio/utt2spk > fixed_utt2spk

#mgb3_hf Data preparation
# if [ $stage -le 1 ];then
#    echo " Stage 1: Data Preparation"
#    local/data_prep.sh $mgb3_root data/mgb3_hf/

# fi

# if [ $stage -le 2 ];then
#    echo " Stage 2: Data Conversion"
#    for part in train dev test
#    do
#       echo ${part}
#       steps_transducer/preprocess_audios_for_nnet_train.sh --nj 1 --cmd "$train_cmd" \
#       --storage_name mgb3-hf-$(date +'%m_%d_%H_%M') --use-bin-vad false \
#       --osr 16000 data/mgb3_hf/${part} data/mgb3_hf/${part}_proc_audio  exp/mgb3_hf/${part}_proc_audio
#       utils/fix_data_dir.sh data/${part}_proc_audio || true
#    done
# fi

# cut -d' ' -f1 --complement  data/mgb3_hf/train_proc_audio/text > data/mgb3_hf/lm/egy_transcript_words.txt






#this is madar preparation
# if [ $stage -le 1 ];then
#     hyp_utils/conda_env.sh \
#     local/prepare_madar_csv.py \
#     --corpus-dir $madar_root \
#     --output-dir data/madar

# fi 


# if [ $stage -le 1 ];then
#     hyp_utils/conda_env.sh \
#     local/prepare_mgb3.py \
#     --corpus-dir $mgb3_root \
#     --output-segments-dir /export/fs05/mkhelfi1/mgb3/mgb3/mgb3/segmented_wav \
#     --output-metadata-dir data/mgb3

# fi
