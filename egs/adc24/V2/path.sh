python -V
export HYP_ROOT=$(readlink -f `pwd -P`/../../..)
export TOOLS_ROOT=$HYP_ROOT/tools
. $TOOLS_ROOT/path.sh

K2_ROOT=/home/mkhelfi1/anaconda3/envs/hyp_adc_py311_cu121/lib/python3.11/site-packages
export PYTHONPATH=$K2_ROOT/k2/python:$PYTHONPATH
export PYTHONPATH=$K2_ROOT/build_debug/lib:$PYTHONPATH
