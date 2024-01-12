# -*- coding: utf-8 -*-
import re
import pandas as pd
import jsonlines


def clean(s):
    s = re.sub('\(\d{1,3}\.\d{1,5}\+\d{1,3}\.\d{1,5}\+\d{1,3}\.\d{1,5}\+\d{1,3}\.\d{1,5}\)', '', s)
    s = re.sub('\(\d{1,3}\.\d{1,5}\+\d{1,3}\.\d{1,5}\+\d{1,3}\.\d{1,5}\)', '', s)
    s = re.sub('\(\d{1,3}\.\d{1,5}\+\d{1,3}\.\d{1,5}\)', '', s)
    s = re.sub('\(\d{1,3}\.\d{1,5}\)', '', s)
    s = re.sub('\(\)', '', s)
    return s.lower()


df_icd = pd.read_csv('../data/code.txt', header=None, names=['code', 'name'], sep='\t')

train = []
df_train = pd.read_csv('../data/chip_cdn_train.csv', sep='\t')
for _, row in df_train.iterrows():
    text = clean(row['原始词'])
    norm_lists = row['标准词'].split('##')
    for norm_list in norm_lists:
        origin = text
        entailment = norm_list
        contradiction = df_icd['name'].sample(1).values[0]
        while contradiction == origin or contradiction == entailment:
            contradiction = df_icd['name'].sample(1).values[0]
        data = {'origin': origin, 'entailment': entailment, 'contradiction': contradiction}
        train.append(data)

with jsonlines.open('../data/train.txt', 'w') as writer:
    writer.write_all(train)

dev = []
df_dev = pd.read_csv('../data/chip_cdn_dev.csv', sep='\t')
for _, row in df_train.iterrows():
    text = clean(row['原始词'])
    norm_lists = row['标准词'].split('##')
    for norm_list in norm_lists:
        origin = text
        entailment = norm_list
        contradiction = df_icd['name'].sample(1).values[0]
        while contradiction == origin or contradiction == entailment:
            contradiction = df_icd['name'].sample(1).values[0]
        data = {'origin': origin, 'entailment': entailment, 'contradiction': contradiction}
        dev.append(data)
new_dev = []
for data in dev:
    data1 = data["origin"] + "||" + data["entailment"] + "||" + "1"
    new_dev.append(data1)
    data1 = data["origin"] + "||" + data["contradiction"] + "||" + "0"
    new_dev.append(data1)
with open('../data/dev.txt', 'w') as f:
    for line in new_dev:
        f.write(line + '\n')
