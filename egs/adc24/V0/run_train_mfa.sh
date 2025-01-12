# #!/bin/bash


. ./cmd.sh
. ./path.sh
set -e

stage=1


if [ $stage -le 2 ]; then
    # Extract xvectors for training 
    
    steps_xvec/extract_xvectors_from_wav.sh \
        --cmd "$xvec_cmd" --nj 100 ${xvec_args} \
        --use-bin-vad false  \
        --random-utt-length true \
        --feat-config $feat_config \
        $nnet data/${name}_proc_audio_no_sil \
        $xvector_dir/${name} \
        data/${name}_proc_no_sil 
    done
fi
