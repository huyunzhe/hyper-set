# ─────────────────────────────────────────────────────────────────────────────
# Single-GPU – CIFAR-10/100, hyper-set, layer=1, recur=12, dim=512
# ─────────────────────────────────────────────────────────────────────────────

# Softmax feedforward
CUDA_VISIBLE_DEVICES='1' nohup \
python main.py \
task=ic \
model.type=hyper-set-alt-ff \
data.dataset=c100 \
model.patch=8 \
model.n_layer=1 \
model.n_recur=12 \
model.n_embd=512 \
model.n_head=8 \
model.mlp_hidden=512 \
model.ff=softmax \
experiment.comment="train ic c100 l1r12 dim512 Softmax ff" \
> train_ic_c100_l1r12_dim512_softmax_ff.log 2>&1 &

# Gated feedforward (Phi as Sigmoid in Table 20)
# Support Phi = sigmoid | relu | tanh | softplus | silu | gelu | elu | elup1
CUDA_VISIBLE_DEVICES='2' nohup \
python main.py \
task=ic \
model.type=hyper-set-alt-ff \
data.dataset=c100 \
model.patch=8 \
model.n_layer=1 \
model.n_recur=12 \
model.n_embd=512 \
model.n_head=8 \
model.mlp_hidden=512 \
model.ff=sigmoid \
experiment.comment="train ic c100 l1r12 dim512 Gated ff" \
> train_ic_c100_l1r12_dim512_gated_ff.log 2>&1 &
