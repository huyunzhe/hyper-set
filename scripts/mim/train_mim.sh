# Single gpu training (recur=12, multiplier=1)
CUDA_VISIBLE_DEVICES='2' nohup \
python main.py \
task=mim \
model.n_recur=24 \
model.multiplier=8 \
training.batch_size=256 \
experiment.comment="train mim" \
> train_mim.log 2>&1 &
