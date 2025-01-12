#!/bin/bash
# Copyright
#                2019   Johns Hopkins University (Author: Jesus Villalba)
#                2024   Johns Hopkins University (Author: Mohammed Akram Khelfi)
# Apache 2.0.
#

. ./cmd.sh
. ./path.sh
set -e

stage=1
ngpu=4
config_file=default_config.sh
interactive=false
num_workers=""
use_tb=false
use_wandb=false



. parse_options.sh || exit 1;
. $config_file
. datapath.sh

list_dir=data/adi17
train_dir=data/adi17/train_proc_audio_no_sil_res
dev_dir=data/adi17/dev_proc_audio_no_sil_res
test_dir=data/adi17/test_proc_audio_no_sil_res


if [ -n "$num_workers" ];then
    extra_args="--data.train.data_loader.num-workers $num_workers"
fi
if [ "$use_tb" == "true" ];then
    extra_args="$extra_args --trainer.use-tensorboard"
fi
if [ "$use_wandb" == "true" ];then
    extra_args="$extra_args --trainer.use-wandb --trainer.wandb.project voxceleb-v1.1 --trainer.wandb.name $nnet_name.$(date -Iminutes)"
fi

if [ "$interactive" == "true" ];then
    export cuda_cmd=run.pl
fi

export CUDA_LAUNCH_BLOCKING=1

echo $nnet_s1_dir
echo $nnet_type
echo $nnet_s1_base_cfg
# train_xvector_from_wav.py
# hyperion-train-wav2vec2xvector
# Network Training

# if [ $stage -le 1 ]; then
  
#   conda activate hyperionnew
#   module load cuda/12.1
#   mkdir -p local/log
#   $cuda_cmd \
#     --gpu $ngpu local/log/train.log \
#     hyp_utils/conda_env.sh --conda-env hyperionnew --num-gpus $ngpu \
#     local/cuda_test.py\

# fi


# if [ $stage -le 1 ]; then
  
#   mkdir -p $nnet_s1_dir/log
#   $cuda_cmd \
#     --gpu $ngpu $nnet_s1_dir/log/train.log \
#     hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
#     train_wav2vec2xvector.py $nnet_type --cfg $nnet_s1_base_cfg $nnet_s1_args $extra_args \
#     --data.train.dataset.recordings-file $train_dir/wav.scp \
#     --data.train.dataset.time-durs-file $train_dir/utt2dur \
#     --data.train.dataset.segments-file $train_dir/utt2lang \
#     --data.train.dataset.class-files $list_dir/Arabic_Dialects_class2int_with_id.csv \
#     --data.val.dataset.recordings-file $dev_dir/wav.scp \
#     --data.val.dataset.time-durs-file $dev_dir/utt2dur \
#     --data.val.dataset.segments-file $dev_dir/utt2lang \
#     --trainer.exp-path $nnet_s1_dir \
#     --num-gpus $ngpu \
  
# fi


# if [ $stage -le 2 ]; then
#   if [ "$use_wandb" == "true" ];then
#     extra_args="$extra_args --trainer.wandb.name $nnet_s2_name.$(date -Iminutes)"
#   fi
#   mkdir -p $nnet_s2_dir/log
#   $cuda_cmd \
#     --gpu $ngpu $nnet_s2_dir/log/train.log \
#     hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
#     finetune_wav2vec2xvector.py $nnet_type --cfg $nnet_s2_base_cfg $nnet_s2_args $extra_args \
#     --data.train.dataset.recordings-file $train_dir/wav.scp \
#     --data.train.dataset.time-durs-file $train_dir/utt2dur \
#     --data.train.dataset.segments-file $train_dir/utt2lang \
#     --data.train.dataset.class-files $list_dir/Arabic_Dialects_class2int_with_id.csv \
#     --data.val.dataset.recordings-file $dev_dir/wav.scp \
#     --data.val.dataset.segments-file $dev_dir/utt2lang \
#     --data.val.dataset.time-durs-file $dev_dir/utt2dur \
#     --in-model-file $nnet_s1 \
#     --trainer.exp-path $nnet_s2_dir \
#     --num-gpus $ngpu \
  
# fi



# Finetune full model
if [ $stage -le 3 ]; then
  if [ "$use_wandb" == "true" ];then
    extra_args="$extra_args --trainer.wandb.name $nnet_s3_name.$(date -Iminutes)"
  fi
  mkdir -p $nnet_s3_dir/log
  $cuda_cmd \
    --gpu $ngpu $nnet_s3_dir/log/train.log \
    hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
    finetune_wav2vec2xvector.py $nnet_type --cfg $nnet_s3_base_cfg $nnet_s3_args $extra_args \
    --data.train.dataset.recordings-file $train_dir/wav.scp \
    --data.train.dataset.time-durs-file $train_dir/utt2dur \
    --data.train.dataset.segments-file $train_dir/utt2lang \
    --data.train.dataset.class-files $list_dir/Arabic_Dialects_class2int_with_id.csv \
    --data.val.dataset.recordings-file $dev_dir/wav.scp \
    --data.val.dataset.segments-file $dev_dir/utt2lang \
    --data.val.dataset.time-durs-file $dev_dir/utt2dur \
    --in-model-file $nnet_s2 \
    --trainer.exp-path $nnet_s3_dir \
    --num-gpus $ngpu \
  
fi


# if [ $stage -le 2 ]; then
#     if [ "$use_wandb" == "true" ];then
# 	extra_args="$extra_args --trainer.wandb.name $nnet_s2_name.$(date -Iminutes)"
#     fi
#     mkdir -p $nnet_s2_dir/log
#     $cuda_cmd \
# 	--gpu $ngpu $nnet_s2_dir/log/train.log \
# 	hyp_utils/conda_env.sh --conda-env $HYP_ENV --num-gpus $ngpu \
# 	finetune_xvector_from_wav.py $nnet_type --cfg $nnet_s2_base_cfg $nnet_s2_args $extra_args \
# 	--data.train.dataset.recordings-file $train_dir/wav.scp \
#     --data.train.dataset.time-durs-file $train_dir/utt2dur \
#     --data.train.dataset.segments-file $train_dir/utt2lang \
#     --data.train.dataset.class-files $list_dir/Arabic_Dialects_class2int_with_id.csv \
#     --data.val.dataset.recordings-file $dev_dir/wav.scp \
#     --data.val.dataset.time-durs-file $dev_dir/utt2dur \
#     --data.val.dataset.segments-file $dev_dir/utt2lang \
#     --in-model-file $nnet_s1 \
# 	--trainer.exp-path $nnet_s2_dir \
# 	--num-gpus $ngpu \

# fi