# acoustic features
feat_config=conf/fbank80_stmn_16k.yaml
feat_type=fbank80_stmn

#vad
vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=adi17

# x-vector cfg

nnet_type=resnet
nnet_stages=1
nnet_s1_base_cfg=conf/train_fwseres2net50s8_xvec_stage1_v1.0.yaml

nnet_name=${feat_type}_fwseres2net50s8_v1.0
nnet_s1_name=$nnet_name.s1
nnet_s1_dir=exp/xvector_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0040.pth


nnet_s2_base_cfg=conf/train_fwseres2net50s8_xvec_stage2_v1.0.yaml
nnet_s2_name=${nnet_name}.s2
nnet_s2_dir=exp/xvector_nnets/$nnet_s2_name
nnet_s2=$nnet_s2_dir/model_ep0007.pth

