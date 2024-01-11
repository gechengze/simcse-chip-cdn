# -*- coding: utf-8 -*-
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
import torch.nn.functional as F
from src.model import SupModel
from loguru import logger


class SupTrainer:
    def __init__(self, device, pretrained_model_path, model_save_path, lr, dropout) -> None:
        self.device = torch.device(device)
        self.pretrained_model_path = pretrained_model_path
        self.model_save_path = model_save_path
        self.best = 0
        self.lr = lr
        self.dropout = dropout
        self.model = SupModel(pretrained_bert_path=self.pretrained_model_path, drop_out=self.dropout).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def simcse_sup_loss(self, y_pred):
        y_true = torch.arange(y_pred.shape[0], device=self.device)
        use_row = torch.where((y_true + 1) % 3 != 0)[0]
        y_true = (use_row - use_row % 3 * 2) + 1
        sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
        sim = sim - torch.eye(y_pred.shape[0], device=self.device) * 1e12
        sim = torch.index_select(sim, 0, use_row)
        sim = sim / 0.05
        loss = F.cross_entropy(sim, y_true)
        return loss

    def train_epoch(self, train_dataloader, dev_dataloader):
        self.model.train()
        total_loss = 0.0
        for batch_idx, source in enumerate(tqdm(train_dataloader), start=1):
            real_batch_num = source.get('input_ids').shape[0]
            input_ids = source.get('input_ids').view(real_batch_num * 3, -1).to(self.device)
            attention_mask = source.get('attention_mask').view(real_batch_num * 3, -1).to(self.device)
            token_type_ids = source.get('token_type_ids').view(real_batch_num * 3, -1).to(self.device)

            out = self.model(input_ids, attention_mask, token_type_ids)
            loss = self.simcse_sup_loss(out)
            total_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        logger.info(f'train loss: {total_loss}')

        corrcoef = self.eval(dev_dataloader)
        if corrcoef > self.best:
            self.best = corrcoef
            torch.save(self.model.state_dict(), f'{self.model_save_path}_{self.best:.4f}.pt')
            logger.info(f"higher corrcoef: {self.best:.4f}, save model")

    def eval(self, dataloader):
        self.model.eval()
        sim_tensor = torch.tensor([], device=self.device)
        label_array = np.array([])
        with torch.no_grad():
            for source, target, label in tqdm(dataloader):
                source_input_ids = source.get('input_ids').squeeze(1).to(self.device)
                source_attention_mask = source.get('attention_mask').squeeze(1).to(self.device)
                source_token_type_ids = source.get('token_type_ids').squeeze(1).to(self.device)
                source_pred = self.model(source_input_ids, source_attention_mask, source_token_type_ids)

                target_input_ids = target.get('input_ids').squeeze(1).to(self.device)
                target_attention_mask = target.get('attention_mask').squeeze(1).to(self.device)
                target_token_type_ids = target.get('token_type_ids').squeeze(1).to(self.device)
                target_pred = self.model(target_input_ids, target_attention_mask, target_token_type_ids)

                sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
                sim_tensor = torch.cat((sim_tensor, sim), dim=0)
                label_array = np.append(label_array, np.array(label))
        corr = spearmanr(label_array, sim_tensor.cpu().numpy()).correlation
        logger.info(f'corr: {corr}')
        return corr

    def train(self, num_epochs, train_dataloader, dev_dataloader):
        for epoch in range(num_epochs):
            logger.info(f'epoch {epoch + 1} of {num_epochs}')
            self.train_epoch(train_dataloader, dev_dataloader)
