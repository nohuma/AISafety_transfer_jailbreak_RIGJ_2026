import random
import torch
import numpy as np
import json
import pandas as pd

from transformers import LlamaForCausalLM, MistralForCausalLM


DATASET_NAME_TO_PATH = {
    "advbench_gjo": "data/Advbench/advbench_gjo.csv",
    "harmbench_gjo" : "data/Harmbench/harmbench_gjo.csv",
    "maliciousinstruct": "data/MaliciousInstruct/malicious_gjo.csv"
}

MODEL_NAME_TO_PATH = {
    "llama2-7b": "/root/autodl-tmp/Llama-2-7b-chat-hf", # meta-llama/Llama-2-7b-chat-hf
    "vicuna-7b": "lmsys/vicuna-7b-v1.5",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "gemma-7b": "google/gemma-7b-it",
    "llama3-8b": "meta-llama/Llama-3-8B-Instruct",
    "llama2-13b-hb": "cais/HarmBench-Llama-2-13b-cls",
    "guanaco-7b": "TheBloke/guanaco-7B-HF"
}

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_goals_and_targets(args):
    train_goals = getattr(args, 'goals', [])
    train_targets = getattr(args, 'targets', [])
    test_goals = getattr(args, 'test_goals', [])
    test_targets = getattr(args, 'test_targets', [])
    offset = getattr(args, 'offset', 0)

    if args.train_dataset:
        dataset_path = DATASET_NAME_TO_PATH[args.train_dataset]
        train_data = pd.read_csv(dataset_path)

        train_goals = train_data["goal"].tolist()[offset : offset + args.n_train_data]
        train_targets = train_data["target"].tolist()[offset: offset + args.n_train_data]
    else:
        raise ValueError("A train dataset is neeeded or a positive number of train examples is needed !")

    if args.test_dataset and args.n_test_data > 0:
        test_data = pd.read_csv(args.test_dataset)
        test_targets = test_data['target'].tolist()[offset:offset+args.n_test_data]
        test_goals = test_data['goal'].tolist()[offset:offset+args.n_test_data]
    elif args.n_test_data > 0: 
        test_targets = train_data['target'].tolist()[offset+args.n_train_data:offset+args.n_train_data+args.n_test_data]
        test_goals = train_data['goal'].tolist()[offset+args.n_train_data:offset+args.n_train_data+args.n_test_data]

    assert len(train_goals) == len(train_targets)
    assert len(test_goals) == len(test_targets)
    print('Loaded {} train goals'.format(len(train_goals)))
    print('Loaded {} test goals'.format(len(test_goals)))
    return train_goals, train_targets, test_goals, test_targets


def get_embedding_matrix(model):
    if isinstance(model, (LlamaForCausalLM, MistralForCausalLM)):
        return model.model.embed_tokens.weight

    raise ValueError(f"Unknown model type: {type(model)}")


def get_illegal_tokens(tokenizer):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    if "Baichuan2" in tokenizer.name_or_path:
        ascii_toks += [i for i in range(101, 1000)]

    ascii_toks = tuple(set(ascii_toks))
    return ascii_toks









