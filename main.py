# -*- coding: utf-8 -*-
import argparse
from transformers import BertTokenizer
from src.trainer import SupTrainer
from src.dataset import load_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--model_path', type=str, default='bert-base-chinese')
    parser.add_argument('--save_path', type=str, default='sup_saved')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--toy', type=bool, default=False)
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    trainer = SupTrainer(args.device, args.model_path, args.save_path, args.lr, args.dropout)

    train_dataloader, dev_dataloader = load_data('data/', tokenizer, batch_size=args.batch_size, max_len=args.max_len, toy=args.toy)
    trainer.train(args.num_epochs, train_dataloader, dev_dataloader)
