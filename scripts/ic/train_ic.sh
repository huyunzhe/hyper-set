#!/usr/bin/env bash
# =============================================================================
# scripts/main_ic.sh
# Image Classification (IC) – Hyper-SET training & evaluation scripts
#
# Model types (model.type):
#   hyper-set          – main variant  
#   hyper-set-lora     – + Depth-wise LoRA 
#   hyper-set-basic    – with extra RMSNorm in time_mlp 
#   hyper-set-alt-attn – alternative attention
#   hyper-set-alt-ff   – alternative feed-forward
#   hyper-set-ss       – fix step-size
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# Single-GPU – CIFAR-10/100, hyper-set, layer=1, recur=12, dim=512
# ─────────────────────────────────────────────────────────────────────────────
# CUDA_VISIBLE_DEVICES='4' nohup \
# python main.py \
# task=ic \
# model.type=hyper-set-basic \
# data.dataset=c100 \
# model.patch=8 \
# model.n_layer=1 \
# model.n_recur=12 \
# model.n_embd=512 \
# model.n_head=8 \
# model.mlp_hidden=512 \
# experiment.comment="train c100 l1r12 dim512" \
# > train_ic_c100_l1r12_dim512.log 2>&1 &

# ─────────────────────────────────────────────────────────────────────────────
# Single-GPU – ImageNet-100/1k, hyper-set, layer=1, recur=12, dim=512
# ─────────────────────────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES='2' nohup \
python main.py \
task=ic \
model.type=hyper-set \
data.dataset=in1k \
model.n_layer=1 \
model.n_recur=12 \
model.n_embd=512 \
model.n_head=8 \
model.mlp_hidden=512 \
training.batch_size=256 \
experiment.comment="train ic in1k l1r12 dim512" \
> train_ic_in1k_l1r12_dim512.log 2>&1 &
