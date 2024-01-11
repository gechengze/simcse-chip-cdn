# -*- coding: utf-8 -*-
import pdb

import torch
import argparse
import torch.nn.functional as F
from src.model import SupModel
from transformers import BertTokenizer


def predict(tokenizer, model, text_a, text_b, device):
    token_a = tokenizer([text_a], max_length=64, truncation=True, padding='max_length', return_tensors='pt')
    token_b = tokenizer([text_b], max_length=64, truncation=True, padding='max_length', return_tensors='pt')
    model.eval()
    source_input_ids = token_a.get('input_ids').squeeze(1).to(device)
    source_attention_mask = token_a.get('attention_mask').squeeze(1).to(device)
    source_token_type_ids = token_a.get('token_type_ids').squeeze(1).to(device)
    source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
    # target        [batch, 1, seq_len] -> [batch, seq_len]
    target_input_ids = token_b.get('input_ids').squeeze(1).to(device)
    target_attention_mask = token_b.get('attention_mask').squeeze(1).to(device)
    target_token_type_ids = token_b.get('token_type_ids').squeeze(1).to(device)
    target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
    pdb.set_trace()
    # concat
    sim = F.cosine_similarity(source_pred, target_pred, dim=-1).item()
    return sim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--ckpt', type=str)
    args = parser.parse_args()

    model = SupModel(args.model_path, 0.3).to(args.device)

    model.load_state_dict(torch.load(args.ckpt))
    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    text_a = '口腔黏膜癌'
    text_b = '口腔粘膜恶性肿瘤'
    sim_score = predict(tokenizer, model, text_a, text_b, args.device)
    print(text_a, text_b, sim_score)

    text_a = '口腔黏膜癌'
    text_b = '高血压'
    sim_score = predict(tokenizer, model, text_a, text_b, args.device)
    print(text_a, text_b, sim_score)
