seq_len=336
model=RWKV7_TS_3

for percent in 100
do
for pred_len in 96 192 336 720
do
for lr in 0.0001
do

python main.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data ett_h \
    --seq_len $seq_len \
    --label_len 168 \
    --pred_len $pred_len \
    --batch_size 256 \
    --lradj type4 \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.5 \
    --d_model 128 \
    --n_heads 2 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 2 \
    --itr 3 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --pretrain 0

done
done
done
python inform.py