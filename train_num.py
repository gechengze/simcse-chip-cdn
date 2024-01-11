# -*- coding: utf-8 -*-
import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 准备训练数据和标签
texts, labels = [], []
data = json.load(open('data/CHIP-CDN-train.json'))
for item in data:
    texts.append(item['text'])
    normalized_result = item['normalized_result']
    num_result = len(normalized_result.split('##'))
    if num_result == 1:
        label = 0
    elif num_result == 2:
        label = 1
    else:
        label = 2

    labels.append(label)

train_texts, valid_texts, train_labels, valid_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 设定设备（CPU或GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = '../roberta-chinese'
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)
model.to(device)

tokenized_train_texts = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
tokenized_valid_texts = tokenizer(valid_texts, padding=True, truncation=True, return_tensors="pt")

train_labels = torch.tensor(train_labels)
valid_labels = torch.tensor(valid_labels)

train_dataset = TensorDataset(tokenized_train_texts['input_ids'], tokenized_train_texts['attention_mask'], train_labels)
valid_dataset = TensorDataset(tokenized_valid_texts['input_ids'], tokenized_valid_texts['attention_mask'], valid_labels)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
best_valid_acc = 0.0  # 记录最佳验证准确率

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # 验证模型
    model.eval()
    valid_preds = []
    valid_labels_list = []

    for batch in valid_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        valid_preds.extend(preds)
        valid_labels_list.extend(labels.cpu().numpy())

    # 计算验证准确率
    valid_acc = accuracy_score(valid_labels_list, valid_preds)
    print(f'Epoch {epoch + 1}, Valid Accuracy: {valid_acc}')

    # 如果验证准确率提高，则保存模型
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        model.save_pretrained('num_saved.pt')
