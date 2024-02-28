work_path=$(dirname $0)
filename=$(basename $work_path)
partition=$1
gpus=$2
datapath=$3
OMP_NUM_THREADS=1 \
srun -p ${partition} -n ${gpus} --ntasks-per-node=${gpus} --cpus-per-task=14 --gres=gpu:${gpus} \
python -u DecisionNCE/main.py \
    --image_path <your image folder path > \
    --meta_file <path for data annotation >