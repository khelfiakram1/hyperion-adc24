# WavLM base trained on 60k LibriLight + 10k GigaSpeech + 24k Voxpopuli + ECAPA-TDNN 512x3

# hugging face model
hf_model_name=wav2vec2xlsr300mar

#vad
# vad_config=conf/vad_16k.yaml

# x-vector training 
nnet_data=data/qasr/train_proc_audio
dev_data=data/qasr/dev_proc_audio
test_data=data/qasr/test_proc_audio

language=ara

bpe_model=data/qasr/ara_lang_bpe_500/bpe.model
# x-vector cfg

nnet_type=hf_wav2vec2rnn_transducer

nnet_s1_base_cfg=conf/train_wav2vec2xlsr300m_transducer_stage1_v3.3.yaml
nnet_s1_args=""

nnet_name=${hf_model_name}_qasr_transducer_500_2GPU_v1.0
nnet_s1_name=$nnet_name.s1

nnet_s1_dir=/export/fs05/mkhelfi1/hyperion/qasr/exp/transducer_nnets/$nnet_s1_name
nnet_s1=$nnet_s1_dir/model_ep0168.pth

nnet_s2_base_cfg=conf/train_wav2vec2xlsr300m_transducer_stage2_v3.3.yaml
nnet_s2_args=""
nnet_s2_name=${nnet_name}.s2
nnet_s2_dir=/export/fs05/mkhelfi1/hyperion/qasr/exp/transducer_nnets/$nnet_s2_name
nnet_s2=$nnet_s2_dir/model_ep0020.pth

echo $nnet_s1_dir