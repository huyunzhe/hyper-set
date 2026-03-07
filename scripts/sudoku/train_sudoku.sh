# Training
CUDA_VISIBLE_DEVICES='5' nohup \
python main.py \
task=sudoku \
experiment.comment="train sudoku" \
> train_sudoku.log 2>&1 &
