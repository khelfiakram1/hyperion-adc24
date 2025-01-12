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
nnet_s1_base_cfg=conf/train_fwseres2net50s8_xvec_stage1_v2.0.yaml

nnet_name=${feat_type}_fwseres2net50s8_v2.0
nnet_s1_name=$nnet_name.s1
nnet_s1_dir=exp/xvector_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0030.pth


nnet_s2_base_cfg=conf/train_fwseres2net50s8_xvec_stage2_v1.0.yaml
nnet_s2_name=${nnet_name}.s2
nnet_s2_dir=exp/xvector_nnets/$nnet_s2_name
nnet_s2=$nnet_s2_dir/model_ep0020.pth

do_pca=true


# back-end
do_plda=true
do_snorm=false
do_qmf=false
do_pca=true


plda_aug_config=conf/speed_perturbation_aug.yaml
plda_num_augs=0
if [ $plda_num_augs -eq 0 ]; then
    plda_data=adi17/train
else
    plda_data=adi17_train_augx${plda_num_augs}
fi
plda_type=splda
lda_dim=200
plda_y_dim=150
plda_z_dim=200
