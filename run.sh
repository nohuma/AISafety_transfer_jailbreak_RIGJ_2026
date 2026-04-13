lr=1
beta_1=0.9
beta_2=0.9999
begin_tau=5
final_tau=1
n_train_data=20
num_steps=1000
num_adv_tokens=100

nohup python -u ngd_main.py \
    --source-model llama2-7b \
    --train-dataset harmbench_gjo \
    --lr $lr \
    --beta-1 $beta_1 \
    --beta-2 $beta_2 \
    --begin-tau $begin_tau \
    --final-tau $final_tau \
    --n-train-data $n_train_data\
    --num-steps $num_steps\
    --num-adv-tokens $num_adv_tokens\
    > ./rigj-harmbench-gjo-vicuna-steps$num_steps-lr$lr-beta$beta_1-$beta_2-tau$begin_tau-$final_tau-trainnum$n_train_data-bf16-num$num_adv_tokens.out 2>&1
