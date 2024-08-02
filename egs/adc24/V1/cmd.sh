
if [ "$(hostname -d)" == "cm.gemini" ];then
    #export train_cmd="queue.pl --config conf/coe_gpu_short.conf --mem 4G"
    export train_cmd="queue.pl --config conf/coe_gpu_long.conf --mem 4G"
    export cuda_cmd="queue.pl --config conf/coe_gpu_long.conf --mem 20G"
    export cuda_cmd="queue.pl --config conf/coe_gpu_rtx.conf --mem 40G"
    #export cuda_cmd="queue.pl --config conf/coe_gpu_v100.conf --mem 20G"
    export cuda_eval_cmd="queue.pl --config conf/coe_gpu_short.conf --mem 4G"
    # export cuda_eval_cmd="queue.pl --config conf/coe_gpu_long.conf --mem 4G"
else
    export train_cmd="slurm.pl --config conf/slurm_clsp.conf --mem 4G"
    #export cuda_cmd="slurm.pl --config conf/slurm_clsp.conf --mem 20G"
    export cuda_cmd="slurm.pl --config conf/slurm_clsp_a100.conf --mem 20G"
    export cuda_eval_cmd="$train_cmd$"
fi

