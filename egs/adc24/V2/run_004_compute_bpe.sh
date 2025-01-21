#!/bin/bash


. ./cmd.sh
. ./path.sh
set -e

vocab_sizes=(
  200
  5000
  8000
)


dl_dir=$PWD/download
stage=1
stop_stage=4
config_file=default_config.sh

. parse_options.sh || exit 1;
. ./datapath.sh 
. $config_file

for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/qasr/ar_lang_bpe_${vocab_size}
    mkdir -p $lang_dir

    echo "<eps> 0" > $lang_dir/words.txt
    echo "!SIL 1" >> $lang_dir/words.txt
    echo "<UNK> 2" >> $lang_dir/words.txt

    num_words=$(cat $lang_dir/words.txt | wc -l)
    echo "<s> ${num_words}" >> $lang_dir/words.txt
    num_words=$(cat $lang_dir/words.txt | wc -l)
    echo "</s> ${num_words}" >> $lang_dir/words.txt
    num_words=$(cat $lang_dir/words.txt | wc -l)
    echo "#0 ${num_words}" >> $lang_dir/words.txt

    

    hyp_utils/conda_env.sh \
    local/train_bpe_model.py \
    --lang-dir $lang_dir \
    --vocab-size $vocab_size \
    --transcript data/qasr/lm/qasr_transcript_words_clean.txt

    echo $K2_ROOT
    echo $HYP_ENV

    mkdir -p $lang_dir/log
    $cuda_cmd \
    --gpu 1 $lang_dir/log/train.log \
    hyp_utils/conda_env.sh \
    local/prepare_lang_bpe.py \
    --lang-dir $lang_dir


    # mkdir -p data/bpe/log
    # $cuda_cmd \
    # --gpu 1 data/bpe/log/train.log \
    # hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus 1 \
    # local/test.py
    # echo $HYP_ENV
    # conda activate $HYP_ENV
    # local/prepare_lang_bpe.py --lang-dir $lang_dir


    # if [ ! -f $lang_dir/L_disambig.pt ]; then
    #     python3 local/prepare_lang_bpe.py --lang-dir $lang_dir
    # fi

    
done




