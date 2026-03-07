# ─────────────────────────────────────────────────────────────────────────────
# Single-GPU – CIFAR-10/100, hyper-set, layer=1, recur=12, dim=512
# ─────────────────────────────────────────────────────────────────────────────

# Sigmoid attention (Bisotfmax -> Sigmoid)
CUDA_VISIBLE_DEVICES='1' nohup \
python main.py \
task=ic \
model.type=hyper-set-alt-attn \
data.dataset=c100 \
model.patch=8 \
model.n_layer=1 \
model.n_recur=12 \
model.n_embd=512 \
model.n_head=8 \
model.mlp_hidden=512 \
model.attention=sigma \
experiment.comment="train ic c100 l1r12 dim512 sigmoid attention" \
> train_ic_c100_l1r12_dim512_sigmoid_attention.log 2>&1 &

# Linear attention (Phi as Sigmoid in Table 19)
# Support Phi = sigmoid | relu | tanh | softplus | silu | gelu | elu | elup1
CUDA_VISIBLE_DEVICES='2' nohup \
python main.py \
task=ic \
model.type=hyper-set-alt-attn \
data.dataset=c100 \
model.patch=8 \
model.n_layer=1 \
model.n_recur=12 \
model.n_embd=512 \
model.n_head=8 \
model.mlp_hidden=512 \
model.attention=sigmoid \
experiment.comment="train ic c100 l1r12 dim512 linear attention" \
> train_ic_c100_l1r12_dim512_linear_attention.log 2>&1 &
