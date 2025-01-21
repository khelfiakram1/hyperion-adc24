
if [ "$(hostname -d)" == "cm.gemini" ];then
    export train_cmd="queue.pl --config conf/coe_gpu_long.conf --mem 4G"
    export cuda_cmd="queue.pl --config conf/coe_gpu_long.conf --mem 20G"
    export cuda_eval_cmd="queue.pl --config conf/coe_gpu_short.conf --mem 4G"
    # export cuda_eval_cmd="queue.pl --config conf/coe_gpu_long.conf --mem 4G"
else
    export train_cmd="slurm.pl --config conf/slurm_clsp.conf --mem 4G --time 512:00:00"
    export cuda_cmd="slurm.pl --config conf/slurm_clsp_a100.conf --mem 20G --time 512:00:00"
    export cuda_eval_cmd="$train_cmd$"
    # export train_cmd="queue.pl --mem 4G -l hostname=\"[bc][01][234589]*\" -V" 
    # export cuda_cmd="queue.pl --mem 20G -l hostname=\"c[01]*\" -V"
    # export cuda_eval_cmd="$train_cmd"
fi

