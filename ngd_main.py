import argparse
import torch 
import datetime                                                   
from ngd_attack import NGAttack                                                                                                                  

from utils import *

def split_by_caret(value):
    return value.split('^')

def get_args():
    parser = argparse.ArgumentParser(description="Configs")
    
    parser.add_argument("--device", type=str, default="cuda:0",help="Device to run the code")
    # dataset params
    parser.add_argument("--train-dataset", "-d", type=str, default="harmbench_gjo", choices=DATASET_NAME_TO_PATH.keys(),help="Selection of malicious question datasets for training.",)
    parser.add_argument("--n-train-data", type=int, default=20, help="End index of the malicious train dataset.")
    parser.add_argument("--test-dataset", type=int, default=None)
    parser.add_argument("--n-test-data", type=int, default=20, help="End index of the malicious test dataset.")
    parser.add_argument("--anchor_datasets",type=str,nargs="+",default=["./data/prompt-driven_benign.txt", "./data/prompt-driven_harmful.txt"],help="Path to the benign dataset used for anchoring harmless direction in PCA space. Should be exactly two datasets: the first one harmless, the second one harmful.")
    # model params
    parser.add_argument("--source-model", type=str, default="llama2-7b",choices = MODEL_NAME_TO_PATH.keys(), help="Selection of the source model.")
    parser.add_argument("--loss-model-path", type=str, default="./anchor_classifier.pth",)
    parser.add_argument("--num-adv-tokens", type=int, default=100,help="Number of the tokens in the adversarial suffix.")
    parser.add_argument("--num-steps", type=int, default=1000,help="Number of the iteration steps.")
    parser.add_argument("--lr", type=float, default=1,)
    parser.add_argument("--beta-1", type=float, default=0.9,)
    parser.add_argument("--beta-2", type=float, default=0.9999,)       
    parser.add_argument("--begin-tau", type=float, default=5.0,)
    parser.add_argument("--final-tau", type=float, default=1.0,)                                                                                                  

    parser.add_argument("--batch-size", "-bz",type=int, default=1, help="Number of the batch size to test.")
    # ===========================================
    parser.add_argument('--save-folder', type=str,default=None, help='Path to save folder.')
    parser.add_argument("--seed", type=int, default=718,help="A fixed random seed.")

    args = parser.parse_args()
    return args


def main(args):
    set_random_seed(args.seed)
    print(f"")
    train_goals, train_targets, test_goals, test_targets = get_goals_and_targets(args)
    print(f"Dataset: {args.train_dataset}")
    print(f"==>> The dataset is loaded.\n")
    source_model, source_tokenizer = get_hfmodel(args.source_model, dtype=torch.bfloat16) 
    print("==>> The source model  is loaded!\n")

    print(f"start time: {datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')}")
    attacker = NGAttack(
        source_model,
        source_tokenizer,
        train_goals,
        train_targets,
        test_goals,
        test_targets,
        args,
    )
    print(attacker)

    attacker.attack()
    print(f"end time: {datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')}")

        
if __name__ == "__main__":
    args = get_args()

    main(args)

