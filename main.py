import argparse
import dataset
import model
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import time
import torch
import train
import transformers
from transformers import get_linear_schedule_with_warmup
import util

def run():
    parser = argparse.ArgumentParser(description='Hugginface Bert for tweet classification')
    parser.add_argument('--start_lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--train_bs', default=8, type=int, help='batch size for training')
    parser.add_argument('--valid_bs', default=4, type=int, help='batch size for validation')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs')
    parser.add_argument('--model_name', type=str, help='bert model name')
    parser.add_argument('--train_file', type=str, help='training file path')
    parser.add_argument('--test_file', type=str, help='testing file path')
    parser.add_argument('--max_len', default=512, type=int, help='token length')
    parser.add_argument('--model_path', type=str, help='Bert Model file path')
    parser.add_argument('--dropout_ratio', default=0.3, type=float, help='dropout ratio')
    parser.add_argument('--num_classes', default=1, type=int, help='number of output classes')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='number of warmup epochs')
    parser.add_argument('--')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    assert(os.path.exists(args.train_file))
    assert(os.path.exists(args.test_file))
    assert(args.warmup_epochs < args.epochs)

    train_df = pd.read_csv(args.train_file)
    test_df = pd.read_csv(args.test_file)

    train_y = train_df['target']

    train_df['keyword'].fillna("none", inplace = True)
    txt = [str(i) + "\n" + str(j) for i, j in zip(train_df['keyword'], train_df['text'])]
    train_df['txt'] = txt

    X_train, X_test, y_train, y_test \
        = train_test_split(train_df['txt'], train_y, random_state=42, test_size=0.2, stratify=train_df.target.values)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    tokenizer = transformers.BertTokenizer.from_pretrained(
        args.model_name,
        do_lower_case=True
    )

    train_dataset = dataset.BertDataset(
        text=X_train.values,
        tokenizer= tokenizer,
        max_len=args.max_len,
        target=y_train.values
    )

    valid_dataset = dataset.BertDataset(
        text=X_test.values,
        tokenizer= tokenizer,
        max_len=args.max_len,
        target=y_test.values
    )

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        args.train_bs,
        shuffle=True,
        num_workers=4
    )

    valid_dl = torch.utils.data.DataLoader(
        valid_dataset,
        args.valid_bs,
        shuffle=True,
        num_workers=1
    )

    bert = model.BertUncased(args.model_name, dp=args.dropout_ratio, num_classes=args.num_classes)
    bert = bert.to(device)

    for param in bert.parameters():
        param.requires_grad = True

    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    params = list(bert.named_parameters())
    modified_params = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optim = torch.optim.AdamW(modified_params, lr=args.start_lr, weight_decay=1e-4)

    total_steps = int(len(train_df) * args.epochs / args.train_bs)
    warmup_steps = int(len(train_df) * args.warmup_epochs / args.train_bs)

    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    train_stat = util.AvgStats()
    valid_stat = util.AvgStats()

    best_acc = 0.0
    best_f1_score = 0.0

    criterion = torch.nn.CrossEntropyLoss()

    print("\nEpoch\tTrain Acc\tTrain Loss\tTrain F1\tValid Acc\tValid Loss\tValid F1")
    for i in range(args.epochs):
        start = time.time()
        losses, ops, targs = train.train(train_dl, bert, criterion, optim, sched, device)
        duration = time.time() - start
        train_acc = accuracy_score(ops, targs)
        train_f1_score = f1_score(ops, targs)
        train_loss = sum(losses)/len(losses)
        train_stat.append(train_loss, train_acc, train_f1_score, duration)
        start = time.time()
        lossesv, opsv, targsv = train.test(valid_dl, bert, criterion, device)
        duration = time.time() - start
        valid_acc = accuracy_score(opsv, targsv)
        valid_f1_score = f1_score(opsv, targsv)
        valid_loss = sum(lossesv)/len(lossesv)
        valid_stat.append(valid_loss, valid_acc, valid_f1_score, duration)
        if valid_acc > best_acc:
            best_acc = valid_acc
            util.save_checkpoint(bert, True, './best_acc_model.pth.tar')
        print("\n{}\t{:06.8f}\t{:06.8f}\t{:06.8f}\t{:06.8f}\t{:06.8f}\t{:06.8f}".format(i+1, train_acc*100, train_loss, 
                                                    train_f1_score, valid_acc*100, 
                                                    valid_loss, valid_f1_score))

    
    # Now Load best model and get predictions
    test_df['keyword'].fillna("none", inplace = True)
    txt = [str(i) + "\n" + str(j) for i, j in zip(test_df['keyword'], test_df['text'])]
    test_df['txt'] = txt

    test_dataset = dataset.BertDataset(
        text=test_df.txt.values,
        tokenizer= tokenizer,
        max_len=args.max_len,
    )

    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        1,
        shuffle=False,
        num_workers=1
    )

    util.load_checkpoint(bert, './best_acc_model.pth.tar')
    lossest, opst, _ = train.test(test_dl, bert, criterion, device)

if __name__ == "__main__":
    run()