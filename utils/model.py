import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .prompt import *
from .common import MODEL_NAME_TO_PATH
import torch.nn as nn
import torch.optim as optim
import numpy as np

def get_hfmodel(model_name, device='cuda', dtype=torch.float16):
    if "cuda" in device:
        device_map = 'auto'
    else:
        device_map = "cpu"

    model_path = MODEL_NAME_TO_PATH[model_name]
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype, 
            low_cpu_mem_usage=True,
            device_map=device_map,
            trust_remote_code=True,
    )
    model.requires_grad_(False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    if "llama-2" in model_path.lower():
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "left"
    if "vicuna" in model_path.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    if 'llama-3' in model_path.lower():
        tokenizer.padding_side = 'left'
    if "mistral" in model_path.lower() or "mixtral" in model_path.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if 'guanaco' in model_path.lower():
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    print("==>> The source model is loaded !\n")
    return model, tokenizer

def get_chat_template(model_name="vicuna"):
    chat_temp_path = f'utils/chat_templates/{model_name}.jinja'
    chat_template = open(chat_temp_path).read()
    chat_template = chat_template.replace('    ', '')
    chat_template = chat_template.replace('}\n', '}')
    return chat_template

class SuffixManager:
    def __init__(self, model_name, tokenizer, user_prompt, target="", num_adv_tokens=20, enable_sys=False, is_prefix=True):
        self.tokenizer = tokenizer
        self.user_prompt = user_prompt
        self.target = target
        self.num_adv_tokens = num_adv_tokens
        self.is_prefix = is_prefix 
        self.adv_content = (self.tokenizer.pad_token * self.num_adv_tokens).strip()

        if 'zephyr' in model_name:
            model_name = 'zephyr'
            system_prompt = ZEPHYR
        elif 'vicuna' in model_name:
            model_name = 'vicuna'
            system_prompt = VICUNA
        elif 'llama3' in model_name:
            model_name = 'llama-3-instruct'
            system_prompt = LLAMA3
        elif 'llama2' in model_name:
            model_name = 'llama-2-chat'
            system_prompt = LLAMA2
        elif 'mistral' in model_name:
            model_name = 'mistral-instruct'
            system_prompt = MISTRAL
        else:
            raise ValueError('model not supported yet')
        
        self.model_name = model_name
        self.system_prompt = system_prompt if enable_sys else ""

    def get_prompt(self, adv_content=None):
        if adv_content is not None:
            self.adv_content = adv_content
            
        if self.is_prefix:
            # prefix
            user_content = self.adv_content + " " + self.user_prompt
        else: 
            # suffix
            user_content = self.user_prompt +" "+ self.adv_content

        messages = [{
            'role': 'system',
            'content': self.system_prompt
        }, {
            'role': 'user',
            'content': user_content
        }, {
            'role': 'assistant',
            'content': self.target
        }]

        self.tokenizer.chat_template = get_chat_template(self.model_name) 

        string = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        string = self.target.join(string.split(self.target)[:-1])
        string = string + self.target

        return string
    
    
    def get_input_ids(self, adv_content=None):
        string = self.get_prompt(adv_content)
        input_ids = self.tokenizer(string, add_special_tokens=False).input_ids

        target_stop = len(input_ids)
        target_start = -1
        adv_start, adv_stop = -1, -1
        
        for i in range(target_stop, 0, -1):
            if self.tokenizer.decode(input_ids[i:]) == self.target:
                target_start = i
                break
        
        if target_start == -1:
            print(string)
            raise ValueError("Target string could not be located in input_ids.")

        search_target = self.adv_content[1:] if self.adv_content.startswith(' ') else self.adv_content
        
        if self.is_prefix:
            # prefix
            for i in range(0, target_start):
                decoded_check = self.tokenizer.decode(input_ids[i:i + self.num_adv_tokens])
                if search_target in decoded_check:
                    adv_start = i
                    adv_stop = i + self.num_adv_tokens
                    break
        else:
            # suffix
            for i in range(target_start, 0, -1):
                if search_target in self.tokenizer.decode(input_ids[i:]):
                    adv_start, adv_stop = i, i + self.num_adv_tokens
                    break
        
        if adv_start == -1:
             raise ValueError("Adversarial content could not be located in input_ids.")

        
        self.adv_slice = slice(adv_start, adv_stop)
        self.target_slice = slice(target_start, target_stop)
        self.logits_slice = slice(target_start - 1, target_stop - 1)
        
        adv = self.tokenizer.decode(input_ids[self.adv_slice])
        response = self.tokenizer.decode(input_ids[self.target_slice])
        
        try:
            assert adv.strip() == self.adv_content.strip() or adv.strip() == self.adv_content[1:].strip() 
        except:
            print(f"adv:{adv}")
            print(f"adv_content:{self.adv_content}")
            raise ValueError()

        assert response == self.target
        input_ids = torch.tensor(input_ids)

        return input_ids
    

def batch_apply_chat_template(tokenizer, texts, model_name):
    if 'zephyr' in model_name:
        model_name = 'zephyr'
        system_prompt = ZEPHYR
    elif 'vicuna' in model_name:
        model_name = 'vicuna'
        system_prompt = VICUNA
    elif 'llama3' in model_name:
        model_name = 'llama-3-instruct'
        system_prompt = LLAMA3
    elif 'llama2' in model_name:
        model_name = 'llama-2-chat'
        system_prompt = LLAMA2
    elif 'mistral' in model_name:
        model_name = 'mistral-instruct'
        system_prompt = MISTRAL
    else:
        raise ValueError('model not supported yet')
    
    full_prompt_list = []
    tokenizer.chat_template = get_chat_template(model_name)
    for idx, text in enumerate(texts):
        messages = [{
            'role': 'system',
            'content': ""
        }, {
            'role': 'user',
            'content': text
        }]

        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_prompt_list.append(full_prompt)

    return full_prompt_list


def get_hidden_states(model, tokenizer, full_prompt_list):
    model.eval()
    hidden_state_list = []

    with torch.no_grad():
        for full_prompt in tqdm(full_prompt_list, desc="Calculating hidden states"):
            inputs = tokenizer(
                full_prompt, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            # Get the last hidden state of the last token for each sequence
            # We use -1 to index the last layer, and -1 again to index the hidden state of the last token
            hidden_state_list.append(outputs.hidden_states[-1][:, -1, :]) 
    hidden_state_list = torch.stack(hidden_state_list)
    return hidden_state_list


class AnchorClassifier(nn.Module):
    """
        Forward pass for the anchor sample classifier.
        Input: 4D reduced representation (Tensor)
        Output: Probability of being 'harmful' (class 1)
    """
    def __init__(self, input_dim=4, output_dim=1):
        super(AnchorClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.linear(x)
        prob = self.sigmoid(logits)
        return prob 

    def train_model(self, benign_data, harmful_data, epochs=1000, lr=1e-3, save_path="anchor_classifier.pth"):
        """
        :param benign_data: shape=(100,4)
        :param harmful_data: shape=(100,4)
        :param epochs
        :param lr
        :param save_path
        """
        X = np.vstack([benign_data, harmful_data])
        y = np.hstack([np.zeros(100), np.ones(100)])
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)

        criterion = nn.BCELoss()  
        optimizer = optim.Adam(self.parameters(), lr=lr)

        best_acc = 0.0
        print(f"Traing begin (device:{self.device})...")
        for epoch in range(epochs):
            self.train()  
            optimizer.zero_grad()
            pred_prob = self(X)  
            loss = criterion(pred_prob, y)
            loss.backward()
            optimizer.step()

            pred_label = (pred_prob > 0.5).float()
            acc = (pred_label == y).sum().item() / len(y)

            if acc > best_acc:
                best_acc = acc
                torch.save(self.state_dict(), save_path)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

        print(f"\nTraining complete! Best Accuracy: {best_acc:.4f}")
        print(f"Model saved to: {save_path}")

    def load_model(self, model_path="anchor_classifier.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        try:
            self.load_state_dict(torch.load(model_path))
        except:
            self.load_state_dict(torch.load(model_path, map_location=self.device))
        self.eval()
        print(f"Model loaded successfully from {model_path}! Device: {self.device}")

    def predict(self, features):
        if not hasattr(self, 'device'):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(self.device)

        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32).to(self.device)
        elif not isinstance(features, torch.Tensor):
            raise ValueError("Input must be np.array or torch.Tensor with dimensions (4,) or (N,4)")
        
        with torch.no_grad():
            pred_probs = self(features).cpu().numpy()
            pred_labels = (pred_probs > 0.5).astype(int)

        if pred_labels.ndim == 2 and pred_labels.shape[1] == 1:
            pred_labels = pred_labels.squeeze(1)
            
        if pred_labels.size == 1:
            pred_labels = pred_labels.item()
            pred_probs = pred_probs.item()

        return pred_labels, pred_probs
