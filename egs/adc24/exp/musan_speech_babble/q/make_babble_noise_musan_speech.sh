#!/bin/bash
cd /home/mkhelfi1/hyperion/egs/adc24
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
hyp_utils/conda_env.sh make-babble-noise-audio-files.py --output-audio-format flac --min-spks 3 --max-spks 10 --num-reuses 5 --write-time-durs data/musan_speech_babble/utt2dur --input data/musan_speech/wav.scp --output-path /home/mkhelfi1/hyperion/egs/adc24/exp/musan_speech_babble --output-script data/musan_speech_babble/wav.scp 
EOF
) >exp/musan_speech_babble/log/make_babble_noise_musan_speech.log
time1=`date +"%s"`
 ( hyp_utils/conda_env.sh make-babble-noise-audio-files.py --output-audio-format flac --min-spks 3 --max-spks 10 --num-reuses 5 --write-time-durs data/musan_speech_babble/utt2dur --input data/musan_speech/wav.scp --output-path /home/mkhelfi1/hyperion/egs/adc24/exp/musan_speech_babble --output-script data/musan_speech_babble/wav.scp  ) 2>>exp/musan_speech_babble/log/make_babble_noise_musan_speech.log >>exp/musan_speech_babble/log/make_babble_noise_musan_speech.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/musan_speech_babble/log/make_babble_noise_musan_speech.log
echo '#' Finished at `date` with status $ret >>exp/musan_speech_babble/log/make_babble_noise_musan_speech.log
[ $ret -eq 137 ] && exit 100;
touch exp/musan_speech_babble/q/sync/done.101974
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o exp/musan_speech_babble/q/make_babble_noise_musan_speech.log -l hostname=[bc][01][234589]* -V -l mem_free=4G,ram_free=4G    /home/mkhelfi1/hyperion/egs/adc24/exp/musan_speech_babble/q/make_babble_noise_musan_speech.sh >>exp/musan_speech_babble/q/make_babble_noise_musan_speech.log 2>&1
