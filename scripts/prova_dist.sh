ngpu=${1:-8}

echo ${@:2}

python -m torch.distributed.launch --nproc_per_node=$ngpu main.py \
	${@:2}