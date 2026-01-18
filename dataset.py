from torch.utils.data import Dataset
import json
import numpy as np
import torch

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0

        self._offsets = [0]
        with open(data_path, "rb") as f:
            while f.readline():
                self._offsets.append(f.tell())
        self._total_lines = len(self._offsets) - 1
    
    def __len__(self):
        return self._total_lines

    def __getitem__(self, index):
        with open(self.data_path, "rb") as f:
            f.seek(self._offsets[index])
            line = f.readline().decode('utf-8')
        sample = json.loads(line)
        text = f'''{self.tokenizer.bos_token}{sample['text']}'''
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        
        text_len = len(input_id)
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        padding_mask = [1] * text_len + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1], dtype=np.int64)
        Y = np.array(input_id[1:], dtype=np.int64)
        padding_mask = np.array(padding_mask[1:], dtype=np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(padding_mask)

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0

        self._offsets = [0]
        with open(data_path, "rb") as f:
            while f.readline():
                self._offsets.append(f.tell())
        self._total_length = len(self._offsets) - 1
    
    def __len__(self):
        return self._total_length

    def __getitem__(self, index):
        with open(self.data_path, "rb") as f:
            f.seek(self._offsets[index])
            line = f.readline().decode("utf-8")
        sample = json.loads(line)
        text = self.tokenizer.apply_chat_template(sample, 
                                                  tokenize=False, 
                                                  add_generation_prompt=False)
        input_id = self.tokenizer(text).data["input_ids"][:self.max_length]

        text_len = len(input_id)
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        loss_mask = self.generate_loss_mask(input_id)

        input_id = np.array(input_id)
        X = np.array(input_id[:-1], dtype=np.int64)
        Y = np.array(input_id[1:], dtype=np.int64)
        loss_mask = np.array(loss_mask[1:], dtype=np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)
    
    def generate_loss_mask(self, input_id):
        mask = [0] * len(input_id)
        a_sequence = self.tokenizer("<|im_start|>assistant\n")['input_ids']
        a_length = len(a_sequence)

        n = len(input_id)
        i = 0

        while i <= n - a_length:
            match_ = True
            for k in range(a_length):
                if a_sequence[k] != input_id[i+k]:
                    match_ = False
                    break
            if match_:
                j = None
                for idx in range(i + a_length, n):
                    if input_id[idx] == self.tokenizer.eos_token_id:
                        j = idx
                        break
                if j is not None:
                    start = i + a_length
                    end = j
                    if start <= end:
                        for pos in range(start, end+1):
                            if pos < len(mask):
                                mask[pos] = 1
                i += a_length
            else:
                i += 1
        return mask
