# AISafety_transfer_jailbreak_RIGJ_2026

This is the official repository for ACL 2026 Main paper "Enhancing the Transferability of Jailbreak Attacks on Large Language Models via Exploiting Reparameterization Invariance" by Ao Wang, Xinghao Yang, Yongshun Gong, Wei Liu, Baodi Liu, and Weifeng Liu*.

## Installation

You can download the packages needed as follow:

```
conda create -n rigj python==3.11
conda activate rigj
pip install -r requirement.txt 
```
## Training / Attack Generation

Run the following command to perform adversarial prompt optimization:
```
nohup python -u ngd_main.py \
    --source-model llama2-7b \
    --train-dataset harmbench_gjo \
    --lr 1 \
    --beta-1 0.9 \
    --beta-2 0.9999 \
    --begin-tau 5 \
    --final-tau 1 \
    --n-train-data 20 \
    --num-steps 1000 \
    --num-adv-tokens 100 \
    > ./rigj-harmbench-gjo-vicuna-steps1000-lr1-beta0.9-0.9999-tau5-1-trainnum20-bf16-num100.out 2>&1
```
or run 
```
sh run.sh
```
## Argument Description
| Argument | Description |
|----------|------------|
| `--source-model` | Source model used for optimization (e.g., LLaMA2-7B) |
| `--train-dataset` | Training dataset (e.g., HarmBench) |
| `--lr` | Learning rate |
| `--beta-1` | First-order momentum coefficient |
| `--beta-2` | Second-order momentum coefficient |
| `--begin-tau` | Initial temperature |
| `--final-tau` | Final temperature |
| `--n-train-data` | Number of training samples |
| `--num-steps` | Number of optimization steps |
| `--num-adv-tokens` | Length of adversarial token sequence |
## Output

Logs will be saved to:
```
./rigj-harmbench-gjo-vicuna-steps1000-lr1-beta0.9-0.9999-tau5-1-trainnum20-bf16-num100.out
```
The log file includes:
```
optimization progress
intermediate adversarial tokens
training statistics
```
Results will be saved to:
```
./results/
```




