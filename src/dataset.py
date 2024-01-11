# -*- coding: utf-8 -*-
import os
import jsonlines
from torch.utils.data import Dataset, DataLoader


def load_data(file_path, tokenizer, batch_size=64, max_len=64, toy=False):
    with jsonlines.open(os.path.join(file_path, 'train.txt'), 'r') as f:
        data = [(line['origin'], line['entailment'], line['contradiction']) for line in f]

    if toy:
        data = data[:128]
    train_dataset = TrainDataset(data, tokenizer, max_len)

    with open(os.path.join(file_path, 'dev.txt'), 'r', encoding='utf8') as f:
        data = []
        for line in f.readlines():
            line = line.replace('\n', '')
            data.append((line.split("||")[0], line.split("||")[1], line.split("||")[2]))
    if toy:
        data = data[:128]
    dev_dataset = TestDataset(data, tokenizer, max_len)
    train_dl = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    dev_dl = DataLoader(dev_dataset, shuffle=False, batch_size=batch_size)

    return train_dl, dev_dl


class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def text2id(self, text):
        text_ids = self.tokenizer([text[0], text[1], text[2]], max_length=self.max_len, truncation=True,
                                  padding='max_length', return_tensors='pt')

        return text_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.text2id(self.data[index])


class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def text2id(self, text):
        text_ids = self.tokenizer(text, max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
        return text_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.text2id(self.data[index][0]), self.text2id(self.data[index][1]), int(self.data[index][2])
