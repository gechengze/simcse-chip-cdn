# -*- coding: utf-8 -*-
import random
import re
import pandas as pd
import jsonlines
import json


mapping_data = json.load(open('/Users/gechengze/project/fdu-paper/simcse-chip-cdn/data/mappings.json'))
all_names = list(mapping_data[0].keys())

train = []
df_train = pd.read_csv('../data/imcs_train.csv', sep='\t')
for _, row in df_train.iterrows():
    text = row['原始词']
    origin = text
    entailment = row['标准词']
    contradiction = random.choice(all_names)
    while contradiction == origin or contradiction == entailment:
        contradiction = random.choice(all_names)
    data = {'origin': origin, 'entailment': entailment, 'contradiction': contradiction}
    train.append(data)

with jsonlines.open('../data/train.txt', 'w') as writer:
    writer.write_all(train)

dev = []
df_dev = pd.read_csv('../data/imcs_dev.csv', sep='\t')
for _, row in df_dev.iterrows():
    text = row['原始词']
    if len(row['标准词'].split('##')) > 1:
        continue

    origin = text
    entailment = row['标准词']
    contradiction = random.choice(all_names)
    while contradiction == origin or contradiction == entailment:
        contradiction = random.choice(all_names)
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
