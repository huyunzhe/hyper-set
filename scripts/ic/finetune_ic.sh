# ─────────────────────────────────────────────────────────────────────────────
# Single-GPU – ImageNet-100/1k, hyper-set, layer=1, recur=12, dim=512
# ─────────────────────────────────────────────────────────────────────────────

CUDA_VISIBLE_DEVICES='2' nohup \
python main.py \
task=ic \
model.type=hyper-set \
data.dataset=c100 \
model.n_layer=12 \
model.n_recur=1 \
model.n_embd=768 \
model.n_head=12 \
model.mlp_hidden=768 \
training.batch_size=256 \
training.epochs=50 \
training.lr=1e-4 \
training.weight_decay=1e-5 \
experiment.finetune=true \
experiment.ckpt_path="./pretrained_weights/ic/IC_ImageNet1k_layer12_recur1_dim768_multiplier1.ckpt" \
experiment.comment="finetune ic c100 l12r1 dim768" \
> finetune_ic_c100_l12r1_dim768.log 2>&1 &
