#!/bin/bash
cd /home/mkhelfi1/hyperionnew/hyperion/egs/adc24/V0
module load cuda/12.1
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
hyp_utils/conda_env.sh --conda-env hyperionnew --num-gpus 2 local/cuda_test.py 
EOF
) >local/log/train.log
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>local/log/train.log
  unset CUDA_VISIBLE_DEVICES
fi
time1=`date +"%s"`
 ( hyp_utils/conda_env.sh --conda-env hyperionnew --num-gpus 2 local/cuda_test.py  ) &>>local/log/train.log
ret=$?
sync || true
time2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>local/log/train.log
echo '#' Accounting: end_time=$time2 >>local/log/train.log
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>local/log/train.log
echo '#' Finished at `date` with status $ret >>local/log/train.log
[ $ret -eq 137 ] && exit 100;
touch local/q/done.1115259
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=ALL --ntasks-per-node=1 --nodes=1  -p gpu-a100 --account=a100acct --gpus=2  --mem-per-cpu 20G  --open-mode=append -e local/q/train.log -o local/q/train.log  /home/mkhelfi1/hyperionnew/hyperion/egs/adc24/V0/local/q/train.sh >>local/q/train.log 2>&1
