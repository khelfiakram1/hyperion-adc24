# Wav2Vec2 Multilingual 300M params

# hugging face model
hf_model_name=wav2vec2xlsr300m

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=adi17

# x-vector cfg

nnet_type=hf_wav2vec2resnet1d

nnet_s1_base_cfg=conf/train_wav2vec2xlsr300m_ecapatdnn512x3_stage1_v2.0.yaml
nnet_s1_args=""

nnet_name=${hf_model_name}_ecapatdnn512x3_v2.0
nnet_s1_name=$nnet_name.s1

nnet_s1_dir=/export/fs05/mkhelfi1/hyperion/adi/exp/xvector_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0035.pth

# nnet_s2_base_cfg=conf/train_wavlmlarge_loraqv_ecapatdnn512x3_stage2_v2.0.yaml
# nnet_s2_args=""
# nnet_s2_name=${nnet_name}.s2
# nnet_s2_dir=exp/xvector_nnets/$nnet_s2_name
# nnet_s2=$nnet_s2_dir/model_ep0009.pth

# nnet_s3_base_cfg=conf/train_wavlmlarge_loraqv_ecapatdnn512x3_stage3_v2.0.yaml
# nnet_s3_args=""
# nnet_s3_name=${nnet_name}.s3
# nnet_s3_dir=exp/xvector_nnets/$nnet_s3_name
# nnet_s3=$nnet_s3_dir/model_ep0004.pth

# back-end
do_plda=false
do_snorm=true
do_qmf=true
do_voxsrc22=true

plda_aug_config=conf/reverb_noise_aug.yaml
plda_num_augs=0
if [ $plda_num_augs -eq 0 ]; then
    plda_data=adi17
else
    plda_data=adi17_augx${plda_num_augs}
fi
plda_type=splda
lda_dim=200
plda_y_dim=150
plda_z_dim=200
