# ─────────────────────────────────────────────────────────────────────────────
# Evaluation – load checkpoint
# ─────────────────────────────────────────────────────────────────────────────

CUDA_VISIBLE_DEVICES='1' nohup \
python main.py \
mode=eval \
task=ic \
model.type=hyper-set \
data.dataset=in1k \
model.n_layer=12 \
model.n_recur=1 \
model.n_embd=768 \
model.n_head=12 \
model.mlp_hidden=768 \
training.eval_batch_size=256 \
experiment.ckpt_path="./pretrained_weights/ic/IC_ImageNet1k_layer12_reucr1_dim768_multiplier1.ckpt" \
experiment.comment="eval ic in1k l12r1 dim 768" \
> eval_ic_in1k_l12r1_dim768.log 2>&1 &
