# Single gpu evaluation
CUDA_VISIBLE_DEVICES='2' nohup \
python main.py \
mode=eval_recon \
task=mim \
training.batch_size=100 \
model.n_recur=24 \
model.multiplier=8 \
evaluation.masked_ratio=0.4 \
experiment.ckpt_path="./pretrained_weights/mim/MIM_ImageNet100_layer1_recur24_dim768_multiplier8.pth" \
experiment.comment="eval mim" \
> eval_mim.log 2>&1 &
