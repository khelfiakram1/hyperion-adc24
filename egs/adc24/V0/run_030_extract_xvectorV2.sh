. ./cmd.sh
. ./path.sh
set -e

stage=2
nnet_stage=1
config_file=default_config.sh
use_gpu=false
do_tsne=false
split_dev=false
xvec_chunk_length=12800
. parse_options.sh || exit 1;
. $config_file

if [ "$use_gpu" == "true" ];then
    xvec_args="--use-gpu true --chunk-length $xvec_chunk_length"
    xvec_cmd="$cuda_eval_cmd --mem 4G"
else
    xvec_cmd="$train_cmd --mem 12G"
fi

if [ $nnet_stages -lt $nnet_stage ];then
    nnet_stage=$nnet_stages
fi

if [ $nnet_stage -eq 1 ];then
  nnet=$nnet_s1
  nnet_name=$nnet_s1_name
elif [ $nnet_stage -eq 2 ];then
  nnet=$nnet_s2
  nnet_name=$nnet_s2_name
elif [ $nnet_stage -eq 3 ];then
  nnet=$nnet_s3
  nnet_name=$nnet_s3_name
elif [ $nnet_stage -eq 4 ];then
  nnet=$nnet_s4
  nnet_name=$nnet_s4_name
elif [ $nnet_stage -eq 5 ];then
  nnet=$nnet_s5
  nnet_name=$nnet_s5_name
elif [ $nnet_stage -eq 6 ];then
  nnet=$nnet_s6
  nnet_name=$nnet_s6_name
fi

xvector_dir=exp/xvectors/$nnet_name


if [ $stage -le 2 ]; then
    # Extract xvectors for training 
    for name in adi17/train
    do
	steps_xvec/extract_xvectors_from_wav.sh \
	    --cmd "$xvec_cmd" --nj 20 ${xvec_args} \
	    --use-bin-vad false  \
	    --random-utt-length true --min-utt-length 300 --max-utt-length 600 \
	    --feat-config $feat_config \
    	    $nnet data/${name}_proc_audio_no_sil \
    	    $xvector_dir/${name} \
	    data/${name}_proc_no_sil 
    done
fi


if [ $stage -le 3 ]; then
    # Extracts x-vectors for dev and eval
    for name in adi17/dev_proc_audio_no_sil  adi17/test_proc_audio_no_sil 
    do

	steps_xvec/extract_xvectors_from_wav.sh \
	    --cmd "$xvec_cmd --mem 6G" --nj 17 ${xvec_args} \
	    --feat-config $feat_config \
	    $nnet data/$name \
	    $xvector_dir/${name}
    done
fi
