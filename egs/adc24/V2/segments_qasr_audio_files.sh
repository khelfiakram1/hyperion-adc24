#!/bin/bash
#
#           2024 Ecole de Technologie Superieure (Mohammed Akram Khelfi)
# Apache 2.0.
set -e



. ./path.sh
. datapath.sh
nj=1
cmd="run.pl"

stage=0
file_format=flac
storage_name=$(date +'%m_%d_%H_%M')

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


segments_file='data/qasr/segments.txt'
audio_path=$qasr_root/wav
sample_rate=16000
output_dir='/export/b14/mkhelfi1/hyp-data/qasr'
output_scp='data/qasr/wav.scp'

for f in $recordings_file $segments_file; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

# need to set variables 



args=""
$cmd data/qasr/log/qasr_segments.log \
    hyp_utils/conda_env.sh \
    local/segment_qasr.py \
    --segments-file $segments_file \
    --audio-path $audio_path \
    --output-dir $output_dir \
    --output-scp $output_scp \
    --samplerate $sample_rate 

echo "Succeded generating segmented audio files"
