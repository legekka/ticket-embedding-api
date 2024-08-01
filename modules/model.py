import os
import time

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class EmbeddingModel:
    def __init__(self, model_name, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        if device == "cuda":
            self.model = self.model.half()
            self.batch_size = 32
        else:
            self.batch_size = -1
        torch.cuda.empty_cache()
    
        
    def get_cls_embeddings(self, sentences):
        all_embeddings = []
        if self.batch_size != -1:
            for i in range(0, len(sentences), self.batch_size):
                start_time = time.time()
                if self.batch_size != -1:
                    batch = sentences[i:i + self.batch_size]
                else:
                    batch = [sentences[i]]

                tokens = self.tokenizer(
                    text=batch,
                    truncation=True,
                    padding="max_length" if self.batch_size != -1 else False,
                    max_length=512,
                    return_tensors="pt"
                )
                tokens = {k: v.to(self.model.device) for k, v in tokens.items()}

                with torch.no_grad():
                    outputs = self.model(**tokens, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                    cls_embeddings = hidden_states[-1][:, 0, :].detach().cpu().numpy()
                    all_embeddings.append(cls_embeddings)
        else:
            for sentence in sentences:
                tokens = self.tokenizer(
                    text=sentence,
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt"
                )
                tokens = {k: v.to(self.model.device) for k, v in tokens.items()}

                with torch.no_grad():
                    outputs = self.model(**tokens, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
                    cls_embeddings = hidden_states[-1][:, 0, :].detach().cpu().numpy()
                    all_embeddings.append(cls_embeddings)

        return np.vstack(all_embeddings)