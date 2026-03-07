# ─────────────────────────────────────────────────────────────────────────────
# Single-GPU – CIFAR-10/100, hyper-set, layer=1, recur=12, dim=512
# ─────────────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES='2' nohup \
python main.py \
task=ic \
model.type=hyper-set-ss \
data.dataset=c10 \
model.patch=8 \
model.n_layer=1 \
model.n_recur=12 \
model.n_embd=512 \
model.n_head=8 \
model.mlp_hidden=512 \
model.step_size=0.1 \
experiment.comment="train ic c10 l1r12 dim512 fix_step_size 0.1" \
> train_ic_c10_l1r12_dim512_fix_step_size_0_1.log 2>&1 &
