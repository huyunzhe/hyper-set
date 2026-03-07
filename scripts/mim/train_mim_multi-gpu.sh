# Multi gpu training (recur=24, multiplier=8)
CUDA_VISIBLE_DEVICES='0,1,2,3' nohup torchrun --standalone --nproc_per_node=gpu \
main.py \
task=mim \
model.n_recur=24 \
model.multiplier=8 \
training.batch_size=64 \
experiment.comment="train mim multi gpu" \
> train_mim_multi_gpu.log 2>&1 &
