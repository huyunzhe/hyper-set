# Evaluation
CUDA_VISIBLE_DEVICES='0' nohup \
python main.py \
task=sudoku \
mode=vis \
experiment.ckpt_path="./pretrained_weights/sudoku/Sudoku_layer1_recur24_dim768_multiplier4.pth" \
experiment.comment="vis sudoku" \
> vis_sudoku.log 2>&1 &
