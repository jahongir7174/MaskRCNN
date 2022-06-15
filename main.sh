CONFIG=$1
GPUS=$2
python3 -m torch.distributed.launch --nproc_per_node=$GPUS main.py $CONFIG ${@:3}