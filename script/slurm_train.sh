work_path=`pwd`
work_path=`basename $work_path`
PYTHONPATH=$PYTHONPATH:../../ GLOG_vmodule=MemcachedClient=-1 \
srun -n4 --gres=gpu:4  --ntasks-per-node=4 --cpus-per-task=14  \
python -u DecisionNCE/main.py