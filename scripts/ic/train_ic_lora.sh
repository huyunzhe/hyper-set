# ─────────────────────────────────────────────────────────────────────────────
# Multi-GPU DDP – ImageNet-100, hyper-set-lora, layer=1, recur=12, dim=512, r=32, alpha=128 (2 GPUs)
# ─────────────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES='1,2' nohup torchrun --standalone --nproc_per_node=gpu \
main.py \
task=ic \
model.type=hyper-set-lora \
data.dataset=in100 \
model.n_layer=1 \
model.n_recur=12 \
model.n_embd=512 \
model.n_head=8 \
model.mlp_hidden=512 \
model.lora_r=32 \
model.lora_alpha=128 \
experiment.comment="train ic in100 l1r12 dim512 lora r32 a128 ddp2" \
> train_ic_in100_l1r12_dim512_lora_r32_alpha128_ddp2.log 2>&1 &
