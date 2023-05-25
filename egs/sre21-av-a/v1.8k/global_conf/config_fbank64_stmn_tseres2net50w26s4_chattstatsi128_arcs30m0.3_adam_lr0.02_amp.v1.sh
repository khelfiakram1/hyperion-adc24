# Time SE Res2Net50 w26s4 x-vector with mixed precision training

# acoustic features
feat_config=conf/fbank64_stmn_8k.yaml
feat_type=fbank64_stmn

#vad
vad_config=conf/vad_8k.yaml

# x-vector training
nnet_data=voxcelebcat_sre_alllangs_mixfs

eff_batch_size=512 # effective batch size
min_chunk=4
max_chunk=4
lr=0.02

nnet_type=resnet
dropout=0
embed_dim=256
se_r=256

s=30
margin_warmup=20
margin=0.3
attstats_inner=128

nnet_base_cfg=conf/train_tseres2net50w26s4_xvec_stage1_v1.0.yaml
nnet_name=${feat_type}_tseres2net50w26s4_r${se_r}_chattstatsi128_e${embed_dim}_arcs${s}m${margin}_do${dropout}_adam_lr${lr}_b${eff_batch_size}_amp.v1
nnet_dir=exp/xvector_nnets/$nnet_name
nnet=$nnet_dir/model_ep0075.pth
nnet=$nnet_dir/swa_model_ep0076.pth

# xvector full net finetuning with out-of-domain
ft_batch_size_1gpu=8
ft_eff_batch_size=128 # effective batch size
ft_min_chunk=10
ft_max_chunk=10
ft_ipe=1
ft_lr=0.01
ft_margin=0.5

ft_nnet_base_cfg=conf/train_tseres2net50w26s4_xvec_stage2_v1.0.yaml
ft_nnet_name=${nnet_name}.ft_${ft_min_chunk}_${ft_max_chunk}_arcm${ft_margin}_sgdcos_lr${ft_lr}_b${ft_eff_batch_size}_amp.v1
ft_nnet_dir=exp/xvector_nnets/$ft_nnet_name
ft_nnet=$ft_nnet_dir/model_ep0007.pth


# back-end
plda_aug_config=conf/reverb_noise_aug.yaml
plda_num_augs=0
if [ $plda_num_augs -eq 0 ]; then
    plda_data=voxceleb2cat_train
else
    plda_data=voxceleb2cat_train_augx${plda_num_augs}
fi
plda_type=splda

