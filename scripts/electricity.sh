seq_len=512
model=RWKV7_TS_3

for pred_len in 96 192 336 720
do
for percent in 10
do

python main.py \
    --root_path ./datasets/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 2048 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 128 \
    --n_heads 2 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 16 \
    --percent $percent \
    --gpt_layer 2 \
    --itr 3 \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --pretrain 0
done
done