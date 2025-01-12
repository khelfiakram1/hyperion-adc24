#!/bin/bash
# Copyright       2018   Johns Hopkins University (Author: Jesus Villalba)
#                
# Apache 2.0.
#
. ./cmd.sh
. ./path.sh
set -e


stage=1
nnet_stage=2
config_file=default_config.sh

. parse_options.sh || exit 1;
. $config_file
. datapath.sh

if [ $nnet_stage -eq 1 ]; then
  nnet=$nnet_s1
  nnet_name=$nnet_s1_name
elif [ $nnet_stage -eq 2 ]; then
  nnet=$nnet_s2
  nnet_name=$nnet_s2_name
fi


plda_label=${plda_type}y${plda_y_dim}_v1
be_name=lda${lda_dim}_${plda_label}_${plda_data}
xvector_dir=exp/xvectors/$nnet_name
be_dir=exp/be/$nnet_name/$be_name
score_dir=exp/scores/$nnet_name/${be_name}
score_plda_dir=$score_dir/plda
score_cosine_dir=exp/scores/$nnet_name/cosine



# if [ "$do_plda" == "true" ];then
#   if [ $stage -le 1 ]; then
#     echo "Train PLDA on Voxceleb2"
#     steps_be/train_be_v1.sh \
#       --cmd "$train_cmd" \
#       --lda_dim $lda_dim \
#       --plda_type $plda_type \
#       --y_dim $plda_y_dim --z_dim $plda_z_dim \
#       $xvector_dir/$plda_data/xvector.scp \
#       data/$plda_data \
#       $be_dir
    
# fi
# fi
echo '9ala3'

if [ "$do_pca" != "true" ];then
  exit 0
fi

pca_var_r=0.99
pca_dim=17
be_name=speedaug

xvector_dir=exp/xvectors/$nnet_name
#be_dir=exp/be/fbank80_stmn_ecapatdnn512x3.adc24.so.s1/$be_name

score_dir=exp/scores/$nnet_name/${be_name}
score_cosine_dir=exp/scores/$nnet_name/$be_name/cosine
score_cosine_snorm_dir=exp/scores/$nnet_name/$be_name/cosine_snorm
score_cosine_qmf_dir=exp/scores/$nnet_name/$be_name/cosine_qmf

be_dir=exp/be/$nnet_name
score_be_dir=$score_dir/pca_r${pca_var_r}${pca_dim}



# #plda
# if [ $stage -le 1 ]; then

#     echo "Train PLDA on Voxceleb2"
#     steps_be/train_be_v1_plda.sh --cmd "$train_cmd" \
# 				--lda_dim $lda_dim \
# 				--plda_type $plda_type \
# 				--y_dim $plda_y_dim --z_dim $plda_z_dim \
# 				$xvector_dir/$plda_data/xvector.scp \
# 				data/$plda_data \
# 				$be_dir &


#     wait

# fi


# score_be_dir=$score_dir/${be_name}/plda
# if [ $stage -le 2 ];then
# 	for name in dev test
#  	do
# 	 	echo $name
# 		echo "Eval Voxceleb 1 with LDA+CentWhiten+LNorm+PLDA"
# 		steps_be/eval_be_v1.sh --cmd "$train_cmd" --plda_type $plda_type \
# 			data/${name}_proc_audio_no_sil/utt2lang\
# 			$xvector_dir/adi17/${name}_proc_audio_no_sil/xvector.scp \
# 			$be_dir/lda_lnorm.h5 \
# 			$be_dir/plda.h5 \
# 			$score_be_dir/adi17__${name}

# 		$train_cmd --mem 10G --num-threads 6 $score_be_dir/log/score_voxceleb1.log \
# 			local/score_voxceleb1.sh data/voxceleb1_test $score_be_dir 

# 		for f in $(ls $score_be_dir/*_results);
# 		do
# 		echo $f
# 		cat $f
# 		echo ""
# 		done
# 	done

# fi


if [ $stage -le 10 ]; then

    echo "Train projection on adi17"
    $train_cmd $be_dir/log/train_be.log \
	    hyp_utils/conda_env.sh \
	    steps_be/train_be_v1.py \
	    --v-file scp:$xvector_dir/adi17/train/xvector.scp \
	    --train-list data/adi17/train/utt2lang \
		--pca.pca-var-r $pca_var_r \
		--pca.pca-dim $pca_dim\
	    --output-dir $be_dir \
	   

fi

echo 'cbn'

if [ $stage -le 11 ];then
	for name in test dev
	do
	echo $name
  	$train_cmd \
			${score_dir}/adi_17_${name}.log \
			hyp_utils/conda_env.sh \
			steps_be/eval_cos_sim.py \
			--v-file scp:$xvector_dir/adi17/${name}_proc_audio_no_sil/xvector.scp \
			--trial-list data/adi17/${name}_proc_audio_no_sil/utt2lang \
			--has-labels \
			--model-dir $be_dir \
			--score-file ${score_dir}adi17__${name}_cores.tsv
	done
fi









#groups

# if [ $stage -le 10 ]; then
#   echo "Train projection on adi17"
#   $train_cmd exp/be/fbank80_stmn_resnet.adc24.so.s1.group7/log/train_be.log \
# 	     hyp_utils/conda_env.sh \
# 	     steps_be/train_be_groups.py \
# 	     --v-file scp:exp/xvectors/fbank80_stmn_fwseres2net50s8_v2.0.s1/adi17/train/xvector.scp \
# 	     --train-list data/adi17/train_proc_audio_no_sil/utt2langcopy \
# 		 --pca.pca-var-r $pca_var_r \
# 	     --output-dir exp/be/fbank80_stmn_resnet.adc24.so.s1.group7/ \
	   

# fi

# if [ $stage -le 11 ]; then
#   echo "Train projection on adi17"
#   $train_cmd exp/be/fbank80_stmn_resnet.adc24.so.s1.group7/log/train_be_2.log \
# 	     hyp_utils/conda_env.sh \
# 	     steps_be/train_be_id_group.py \
# 	     --v-file scp:exp/xvectors/fbank80_stmn_fwseres2net50s8_v2.0.s1/adi17/train/xvector.scp \
# 	     --train-list data/adi17/train_proc_audio_no_sil/utt2lang \
# 		 --pca.pca-var-r $pca_var_r \
# 	     --output-dir exp/be/fbank80_stmn_resnet.adc24.so.s1.group7/ \
	   

# fi


# if [ $stage -le 12 ];then
# 	for name in dev_proc_audio_no_sil test_proc_audio_no_sil
# 	do
# 	echo "${name} Evaluation"
#   	$train_cmd \
# 			${score_dir}_p12_groups/group7/${name}.log \
# 			hyp_utils/conda_env.sh \
# 			steps_be/eval_be_v2.py \
# 			--v-file scp:exp/xvectors/fbank80_stmn_fwseres2net50s8_v2.0.s1/adi17/${name}/xvector.scp \
# 			--trial-list data/adi17/${name}/utt2lang \
# 			--model-dir exp/be/fbank80_stmn_resnet.adc24.so.s1.group7/ \
# 			--score-file ${score_dir}_adi17__${name}_cores_so_ep30_geroup7.tsv
# 	done
# fi

# if [ $stage -le 11 ];then
# 	for name in dev_proc_audio_no_sil test_proc_audio_no_sil
# 	do
# 	echo "Dev Evaluation"
#   	$train_cmd \
# 			${score_dir}_p12_groupst/${name}.log \
# 			hyp_utils/conda_env.sh \
# 			steps_be/eval_be_cos.py \
# 			--v-file scp:exp/xvectors/fbank80_stmn_fwseres2net50s8_v2.0.s1/adi17/${name}/xvector.scp \
# 			--trial-list data/adi17/${name}/utt2langcopy \
# 			--model-dir exp/be/fbank80_stmn_fwseres2net50s8_v2.0.s1.groups/ \
# 			--score-file ${score_dir}_adi17__${name}_cores_so_ep30_geroups.tsv
# 	done
# fi


# if [ $stage -le 12 ];then
	
# 	echo "Test Evaluation <5s"
  	# $train_cmd \
	# 		${score_dir}_resnet/test_5s.log \
	# 		hyp_utils/conda_env.sh \
	# 		steps_be/eval_be_v2.py \
	# 		--v-file scp:exp/xvectors/fbank80_stmn_fwseres2net50s8_v2.0.s1/adi17/test_proc_audio_no_sil/xvector.scp \
	# 		--trial-list data/adi17/test_proc_audio_no_sil/utt2lang \
	# 		--dur-file data/adi17/test_proc_audio_no_sil/utt2dur \
	# 		--model-dir exp/be/fbank80_stmn_fwseres2net50s8_v2.0.s1/ \
	# 		--min-dur 0.0 \
	# 		--max-dur 5.0 \
	# 		--score-file ${score_dir}_adi17__test_proc_audio_no_sil_cores_so_ep10_5s.tsv

	# $train_cmd \
	# 		${score_dir}_resnet/test_5s20s.log \
	# 		hyp_utils/conda_env.sh \
	# 		steps_be/eval_be_v2.py \
	# 		--v-file scp:exp/xvectors/fbank80_stmn_fwseres2net50s8_v2.0.s1/adi17/test_proc_audio_no_sil/xvector.scp \
	# 		--trial-list data/adi17/test_proc_audio_no_sil/utt2lang \
	# 		--dur-file data/adi17/test_proc_audio_no_sil/utt2dur \
	# 		--model-dir exp/be/fbank80_stmn_fwseres2net50s8_v2.0.s1/ \
	# 		--min-dur 5.0 \
	# 		--max-dur 20.0 \
	# 		--score-file ${score_dir}_adi17__test_proc_audio_no_sil_cores_so_ep10_5s20s.tsv
	
	# $train_cmd \
	# 		${score_dir}_resnet/test_20s.log \
	# 		hyp_utils/conda_env.sh \
	# 		steps_be/eval_be_v2.py \
	# 		--v-file scp:exp/xvectors/fbank80_stmn_fwseres2net50s8_v2.0.s1/adi17/test_proc_audio_no_sil/xvector.scp \
	# 		--trial-list data/adi17/test_proc_audio_no_sil/utt2lang \
	# 		--dur-file data/adi17/test_proc_audio_no_sil/utt2dur \
	# 		--model-dir exp/be/fbank80_stmn_fwseres2net50s8_v2.0.s1/ \
	# 		--min-dur 20.0 \
	# 		--max-dur 30.0 \
	# 		--score-file ${score_dir}_adi17__test_proc_audio_no_sil_cores_so_ep10_20s.tsv
	
# fi
