# -*- coding: utf-8 -*-
from fastbm25 import fastbm25
import pandas as pd


def bm25_eval():
    df = pd.read_csv('../data/code.txt', header=None, names=['code', 'name'], sep='\t')
    corpus = list(df['name'].values)
    model = fastbm25(corpus)

    df_test = pd.read_csv('../data/chip_cdn_test.csv', sep='\t')

    for k in [1, 5, 10, 20, 30, 50, 100]:
        cnt = 0
        correct = 0
        for _, row in df_test.iterrows():
            if len(row['标准词'].split('##')) > 1:
                continue
            query = row['原始词']
            label = row['标准词']
            result = model.top_k_sentence(query, k=k)
            result = [x[0] for x in result]
            cnt += 1
            if label in result:
                correct += 1

        print(k, correct, cnt, correct / cnt)
