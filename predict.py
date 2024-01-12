# -*- coding: utf-8 -*-
import pdb
from loguru import logger
import pandas as pd
import torch
from fastbm25 import fastbm25
import numpy as np
from tqdm import tqdm
from src.model import SupModel
import torch.nn.functional as F
from transformers import BertTokenizer
import pickle

model_path = 'roberta-chinese'
ckpt = 'sup_saved.pt'
device = torch.device('cuda:0')
model = SupModel(model_path, 0.3).to(device)
model.load_state_dict(torch.load(ckpt))
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()


def get_embedding(text):
    with torch.no_grad():
        token = tokenizer([text], max_length=64, truncation=True, padding='max_length', return_tensors='pt')
        input_ids = token.get('input_ids').squeeze(1).to(device)
        attention_mask = token.get('attention_mask').squeeze(1).to(device)
        token_type_ids = token.get('token_type_ids').squeeze(1).to(device)
        embedding = model(input_ids, attention_mask, token_type_ids)
    return embedding


# embedding_dict = {}
# df = pd.read_csv('data/code.txt', header=None, names=['code', 'name'], sep='\t')
# for name in df['name']:
#     embedding_dict[name] = get_embedding(name)
# with open(f'embedding.pickle', 'wb') as f:
#     pickle.dump(embedding_dict, f)

# embedding_dict = {}
# df = pd.read_csv('data/chip_cdn_test.csv', sep='\t')
# for name in df['原始词']:
#     embedding_dict[name] = get_embedding(name)
# with open(f'embedding_origin.pickle', 'wb') as f:
#     pickle.dump(embedding_dict, f)


with open('embedding_origin.pickle', 'rb') as f:
    origin_embedding = pickle.load(f)

with open('embedding.pickle', 'rb') as f:
    candidate_embedding = pickle.load(f)


k = 30
df = pd.read_csv('data/code.txt', header=None, names=['code', 'name'], sep='\t')
corpus = list(df['name'].values)
bm25 = fastbm25(corpus)

df_test = pd.read_csv('data/chip_cdn_test.csv', sep='\t')
origins, labels, preds = [], [], []
for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
    if len(row['标准词'].split('##')) > 1:
        continue
    query = row['原始词']
    q_embedding = origin_embedding[query]
    label = row['标准词']
    candidate = bm25.top_k_sentence(query, k=k)
    candidate = [x[0] for x in candidate]
    # if label not in candidate:
    #     candidate = candidate[:-1] + [label]
    best_sim = 0
    pred = ''
    for c in candidate:
        c_embedding = candidate_embedding[c]
        sim = F.cosine_similarity(q_embedding, c_embedding, dim=-1).item()
        if sim > best_sim:
            best_sim = sim
            pred = c
    origins.append(query)
    labels.append(label)
    preds.append(pred)

df_result = pd.DataFrame({'原始词': origins, '标准词': labels, '预测结果': preds})
df_result['是否正确'] = df_result['标准词'] == df_result['预测结果']
df_result.to_excel(f'预测结果-top{k}.xlsx', index=False)

