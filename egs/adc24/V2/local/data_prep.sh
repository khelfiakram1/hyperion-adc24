#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh


src=$1
dst=$2

if [ ! -d "$src" ]; then
    mkdir -p "$src/clips"

    hyp_utils/conda_env.sh \
    # local/prepare_data.py \
    # --output-dir $src 
    
fi

hyp_utils/conda_env.sh \
    local/prepare_data.py \
    --output-dir $src 




hyp_utils/conda_env.sh \
    local/create_manifest.py \
    --src-dir $src\
    --dst-dir $dst



hyp_utils/conda_env.sh \
    local/create_splits.py \
    --recordings-path $dst/recordings.jsonl.gz \
    --supervisions-path $dst/supervisions.jsonl.gz\
    --output-dir $dst


for part in dev test train
do
    lhotse kaldi export ${dst}/recordings_${part}.jsonl.gz ${dst}/supervisions_${part}.jsonl.gz ${dst}/${part}
    utils/utt2spk_to_spk2utt.pl ${dst}/${part}/utt2spk > ${dst}/${part}/spk2utt
    utils/fix_data_dir.sh ${dst}/${part}
done

