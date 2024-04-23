#!/bin/bash
cd /home/mkhelfi1/hyperion/egs/adc24
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
hyp_utils/conda_env.sh preprocess-audio-files.py --output-audio-format flac --remove-dc-offset --write-time-durs /home/mkhelfi1/hyperion/egs/adc24/exp/musan_music_proc_audio/utt2dur.musan_music.${SGE_TASK_ID} --part-idx ${SGE_TASK_ID} --num-parts 10 --input data/musan_music/wav.scp --output-path /home/mkhelfi1/hyperion/egs/adc24/exp/musan_music_proc_audio --output-script /home/mkhelfi1/hyperion/egs/adc24/exp/musan_music_proc_audio/wav.musan_music.${SGE_TASK_ID}.scp 
EOF
) >exp/musan_music_proc_audio/log/preproc_audios_musan_music.$SGE_TASK_ID.log
time1=`date +"%s"`
 ( hyp_utils/conda_env.sh preprocess-audio-files.py --output-audio-format flac --remove-dc-offset --write-time-durs /home/mkhelfi1/hyperion/egs/adc24/exp/musan_music_proc_audio/utt2dur.musan_music.${SGE_TASK_ID} --part-idx ${SGE_TASK_ID} --num-parts 10 --input data/musan_music/wav.scp --output-path /home/mkhelfi1/hyperion/egs/adc24/exp/musan_music_proc_audio --output-script /home/mkhelfi1/hyperion/egs/adc24/exp/musan_music_proc_audio/wav.musan_music.${SGE_TASK_ID}.scp  ) 2>>exp/musan_music_proc_audio/log/preproc_audios_musan_music.$SGE_TASK_ID.log >>exp/musan_music_proc_audio/log/preproc_audios_musan_music.$SGE_TASK_ID.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/musan_music_proc_audio/log/preproc_audios_musan_music.$SGE_TASK_ID.log
echo '#' Finished at `date` with status $ret >>exp/musan_music_proc_audio/log/preproc_audios_musan_music.$SGE_TASK_ID.log
[ $ret -eq 137 ] && exit 100;
touch exp/musan_music_proc_audio/q/sync/done.101364.$SGE_TASK_ID
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o exp/musan_music_proc_audio/q/preproc_audios_musan_music.log -l hostname=[bc][01][234589]* -V  -l mem_free=4G,ram_free=4G  -t 1:10 /home/mkhelfi1/hyperion/egs/adc24/exp/musan_music_proc_audio/q/preproc_audios_musan_music.sh >>exp/musan_music_proc_audio/q/preproc_audios_musan_music.log 2>&1
