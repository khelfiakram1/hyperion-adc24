#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh

src=$1
dst=$2
audiodst=$3

# hyp_utils/conda_env.sh \
#     local/Prepare_Qasr_Manifest.py \
#     --src-dir $src \
#     --seg-out-dir $audiodst \
#     --dst-dir $dst

# hyp_utils/conda_env.sh \
#     local/create_splits.py \
#     --recordings-path $dst/recordings.jsonl.gz \
#     --supervisions-path $dst/supervisions.jsonl.gz\
#     --output-dir $dst


for part in dev test train
do
    lhotse kaldi export ${dst}/recordings_${part}.jsonl.gz ${dst}/supervisions_${part}.jsonl.gz ${dst}/${part}
    utils/utt2spk_to_spk2utt.pl ${dst}/${part}/utt2spk > ${dst}/${part}/spk2utt
    utils/fix_data_dir.sh ${dst}/${part}
done

