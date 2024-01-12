# -*- coding: utf-8 -*-
import pdb

import pandas as pd
import torch
import pickle
import numpy as np
from tqdm import tqdm
from src.model import SupModel
import faiss
from transformers import BertTokenizer




def get_embedding():
    model_path = 'roberta-chinese'
    ckpt = 'sup_saved_0.8659.pt'
    device = torch.device('cuda:0')

    model = SupModel(model_path, 0.3).to(device)
    model.load_state_dict(torch.load(ckpt))
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model.eval()

    df_icd = pd.read_csv('data/code.txt', header=None, names=['code', 'name'], sep='\t')
    embedding_dict = {}
    with torch.no_grad():
        for text in tqdm(list(df_icd['name'].values)):
            token = tokenizer([text], max_length=64, truncation=True, padding='max_length', return_tensors='pt')
            input_ids = token.get('input_ids').squeeze(1).to(device)
            attention_mask = token.get('attention_mask').squeeze(1).to(device)
            token_type_ids = token.get('token_type_ids').squeeze(1).to(device)
            embedding = model(input_ids, attention_mask, token_type_ids)
            embedding = embedding[0, ].cpu().numpy().astype(np.float32)
            embedding_dict[text] = embedding
            del input_ids, attention_mask, token_type_ids, embedding

    with open(f'embedding.pickle', 'wb') as f:
        pickle.dump(embedding_dict, f)


def predict():
    with open('embedding.pickle', 'rb') as f:
        embedding_dict = pickle.load(f)
    embeddings = np.vstack(list(embedding_dict.values()))
    faiss.normalize_L2(embeddings)
    _, hidden_size = embeddings.shape
    index = faiss.IndexFlatIP(hidden_size)
    index.add(embeddings)
    texts = list(embedding_dict.keys())

    df = pd.read_csv('data/chip_cdn_test.csv', sep='\t')
    model_path = 'roberta-chinese'
    ckpt = 'sup_saved_0.8659.pt'
    device = torch.device('cuda:0')

    model = SupModel(model_path, 0.3).to(device)
    model.load_state_dict(torch.load(ckpt))
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model.eval()
    with torch.no_grad():
        for text in tqdm(list(df['原始词'].values)):
            token = tokenizer([text], max_length=64, truncation=True, padding='max_length', return_tensors='pt')
            input_ids = token.get('input_ids').squeeze(1).to(device)
            attention_mask = token.get('attention_mask').squeeze(1).to(device)
            token_type_ids = token.get('token_type_ids').squeeze(1).to(device)
            query = model(input_ids, attention_mask, token_type_ids)
            faiss.normalize_L2(query)
            sims, most_sim_idx = index.search(query, 1)  # 选最相似的一个结果
            sims = sims.flatten().tolist()
            most_sim_idx = most_sim_idx.flatten().tolist()
            most_sim_keys = [texts[i] for i in most_sim_idx]
            pdb.set_trace()


# get_embedding()
predict()
