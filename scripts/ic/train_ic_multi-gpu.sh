# ─────────────────────────────────────────────────────────────────────────────
# Multi-GPU DDP – ImageNet-100/1k, hyper-set, layer=12, recur=1, dim=512 (2 GPUs)
# ─────────────────────────────────────────────────────────────────────────────

CUDA_VISIBLE_DEVICES='1,2' nohup torchrun --standalone --nproc_per_node=gpu \
main.py \
task=ic \
model.type=hyper-set \
data.dataset=in1k \
model.n_layer=12 \
model.n_recur=1 \
model.n_embd=512 \
model.n_head=8 \
model.mlp_hidden=512 \
experiment.comment="train ic in1k l12r1 dim512 ddp2" \
> train_ic_in1k_l12r1_dim512_ddp2.log 2>&1 &
