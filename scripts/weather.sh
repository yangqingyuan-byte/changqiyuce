seq_len=512
model=RWKV7_TS_3

for percent in 100
do
for pred_len in 96 192 336 720
do

python main.py \
    --root_path ./datasets/weather/ \
    --data_path weather.csv \
    --model_id weather_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 512 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --decay_fac 0.9 \
    --d_model 128 \
    --n_heads 2 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --lradj type3 \
    --patch_size 8 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 2 \
    --itr 3 \
    --model $model \
    --pretrain 0
    
done
done
python inform.py