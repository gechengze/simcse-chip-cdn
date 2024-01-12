# -*- coding: utf-8 -*-
import pdb

import pandas as pd
import torch
import pickle
import torch.nn.functional as F
from src.model import SupModel
from transformers import BertTokenizer

model_path = 'roberta-chinese'
ckpt = 'sup_saved_0.8659.pt'
device = torch.device('cuda:0')

model = SupModel(model_path, 0.3).to(device)
model.load_state_dict(torch.load(ckpt))
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

df = pd.read_csv('data/chip_cdn_test.csv', sep='\t')
embedding_dict = {}
for text in df['原始词']:
    token = tokenizer([text], max_length=64, truncation=True, padding='max_length', return_tensors='pt')
    input_ids = token.get('input_ids').squeeze(1).to(device)
    attention_mask = token.get('attention_mask').squeeze(1).to(device)
    token_type_ids = token.get('token_type_ids').squeeze(1).to(device)
    embedding = model(input_ids, attention_mask, token_type_ids)
    embedding_dict[text] = embedding

with open(f'embedding.pickle', 'wb') as f:
    pickle.dump(embedding_dict, f)
# sim = F.cosine_similarity(source_pred, target_pred, dim=-1).item()

