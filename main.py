import argparse
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import time
import torch
import transformers
import wandb
from transformers import BertConfig
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import model
import preprocess
import train
import util


def log_wandb(wandb, targets, preds, vtargets, vpreds, train_stats, valid_stats, i):
    tacc0 = len([k for j, k in enumerate(preds) if k == 0 and targets[j] == 0]) * 100.0 / len([k for k in targets if k == 0])
    tacc1 = len([k for j, k in enumerate(preds) if k == 1 and targets[j] == 1]) * 100.0 / len([k for k in targets if k == 1])
    wandb.log({"Training Accuracy": train_stats.accs[i] * 100})
    wandb.log({"Training Accuracy class 0": tacc0})
    wandb.log({"Training Accuracy class 1": tacc1})

    vacc0 = len([k for j, k in enumerate(vpreds) if k == 0 and vtargets[j] == 0]) * 100.0 / len([k for k in vtargets if k == 0])
    vacc1 = len([k for j, k in enumerate(vpreds) if k == 1 and vtargets[j] == 1]) * 100.0 / len([k for k in vtargets if k == 1])
    wandb.log({"Validation Accuracy": valid_stats.accs[i] * 100})
    wandb.log({"Validation Accuracy class 0": vacc0})
    wandb.log({"Validation Accuracy class 1": vacc1})

    wandb.log({"Training Loss": train_stats.losses[i]})
    wandb.log({"Validation Loss": valid_stats.losses[i]})

def is_true(ip):
    return str(ip).lower() == 'true'

def run():
    parser = argparse.ArgumentParser(description='Tweet Classification')
    parser.add_argument('--freeze', default=True, type=is_true, help='If true all layers other than top linear layers will be freezed')
    parser.add_argument('--save_plot', default=False, type=is_true, help='Save loss and accuracy plots')
    parser.add_argument('--track', default=False, type=is_true, help='Track the stats using wandb.')
    parser.add_argument('--wandb_project_name', type=str, help="Name of wandb project.")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    configs = config.get_config()

    train_file = configs['train_file'][1:-1]
    test_file = configs['test_file'][1:-1]
    warmup_epochs = int(configs['warmup_epochs'])
    epochs = int(configs['epochs'])
    model_name = configs['model_name'][1:-1]
    test_size = float(configs['test_size'])
    dropout = float(configs['dropout_ratio'])
    num_classes = int(configs['num_classes'])
    linear_in = int(configs['linear_in'])
    max_len = int(configs['max_len'])
    train_bs = int(configs['train_bs'])
    valid_bs = int(configs['valid_bs'])
    start_lr = float(configs['start_lr'])

    SEED = int(configs['seed'])
    util.set_seed(SEED)

    assert(os.path.exists(train_file))
    assert(os.path.exists(test_file))
    assert(warmup_epochs < epochs)

    if args.track:
        if not 'WANDB_API_KEY' in os.environ:
            raise Exception('wandb api key is not set in th enviornment.')

        os.environ['WANDB_NAME'] = 'hf_' + str(model_name) + "_" + str(int(time.time()))
        wandb.init(project=args.wandb_project_name)
    
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Fix some targets
    ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]
    train_df = preprocess.fix_erraneous(train_df, ids_with_target_error, 0)

    # Remove Duplicates
    #train_df.drop_duplicates(['keyword', 'text', 'target'], keep='first', inplace=True)

    if train_df[train_df['target'] == 0].shape[0] > train_df[train_df['target'] == 1].shape[0]:
        count = train_df[train_df['target'] == 0].shape[0] - train_df[train_df['target'] == 1].shape[0]
        df_sampled = train_df[train_df['target'] == 0].sample(n=count, random_state=SEED)
    elif train_df[train_df['target'] == 1].shape[0] > train_df[train_df['target'] == 0].shape[0]:
        count = train_df[train_df['target'] == 1].shape[0] - train_df[train_df['target'] == 0].shape[0]
        df_sampled = train_df[train_df['target'] == 1].sample(n=count, random_state=SEED)

    for id in df_sampled['id']:
        train_df.drop(train_df[train_df['id'] == id].index[0], inplace=True)

    print(train_df[train_df['target'] == 0].shape[0])
    print(train_df[train_df['target'] == 1].shape[0])
    #train_df = pd.concat([train_df, df_minority_upsampled], axis=0)

    train_y = train_df['target']
    train_df.drop(['target'], axis=1, inplace=True)

    # Concat train and test df to preprocess at the same time
    train_idx = len(train_df)

    total_df = train_df.append(test_df, ignore_index=True)
    total_df = preprocess.preprocess(total_df)

    train_df = total_df[:train_idx]
    test_df = total_df[train_idx:]

    X_train, X_test, y_train, y_test \
        = train_test_split(train_df['text'], train_y, random_state=SEED, test_size=test_size, stratify=train_y.values)  

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    tokenizer = transformers.BertTokenizer.from_pretrained(
        model_name,
        do_lower_case=True
    )

    train_dataset = dataset.BertDataset(
        text=X_train.values,
        tokenizer= tokenizer,
        max_len= max_len,
        target=y_train.values
    )

    valid_dataset = dataset.BertDataset(
        text=X_test.values,
        tokenizer= tokenizer,
        max_len= max_len,
        target=y_test.values
    )

    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        train_bs,
        shuffle=True,
        num_workers=4
    )

    valid_dl = torch.utils.data.DataLoader(
        valid_dataset,
        valid_bs,
        shuffle=True,
        num_workers=1
    )

    bert = model.BertHf(model_name, dp=dropout, num_classes=num_classes, linear_in=linear_in)
    bert = bert.to(device)

    if args.freeze:
        for param in bert.parameters():
            param.requires_grad = False
        
        for param in bert.out.parameters():
            param.requires_grad = True

    else:
        for param in bert.parameters():
            param.requires_grad = True

    if args.track:
        wandb.watch(bert)

    for param in bert.parameters():
        param.requires_grad = True

    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    params = list(bert.named_parameters())
    modified_params = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.1},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optim = torch.optim.AdamW(modified_params, lr=start_lr, eps=1e-8)

    total_steps = int(len(train_df) * epochs / train_bs)
    warmup_steps = int(len(train_df) * warmup_epochs / train_bs)

    if is_true(configs['use_sched']):
        sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    else:
        sched = None

    train_stat = util.AvgStats()
    valid_stat = util.AvgStats()

    best_acc = 0.0
    best_model_file = str(model_name) + '_best.pth.tar'

    criterion = torch.nn.CrossEntropyLoss()

    print("\nEpoch\tTrain Acc\tTrain Loss\tTrain F1\tValid Acc\tValid Loss\tValid F1")
    for i in range(epochs):
        start = time.time()
        losses, ops, targs = train.train(train_dl, bert, criterion, optim, sched, device)
        duration = time.time() - start
        train_acc = accuracy_score(targs, ops)
        train_f1_score = f1_score(targs, ops)
        train_loss = sum(losses)/len(losses)
        train_prec = precision_score(targs, ops)
        train_rec = recall_score(targs, ops)
        train_roc_auc = roc_auc_score(targs, ops)
        train_stat.append(train_loss, train_acc, train_f1_score, train_prec, train_rec, train_roc_auc, duration)
        start = time.time()
        lossesv, opsv, targsv = train.test(valid_dl, bert, criterion, device)
        duration = time.time() - start
        valid_acc = accuracy_score(targsv, opsv)
        valid_f1_score = f1_score(targsv, opsv)
        valid_loss = sum(lossesv)/len(lossesv)
        valid_prec = precision_score(targsv, opsv)
        valid_rec = recall_score(targsv, opsv)
        valid_roc_auc = roc_auc_score(targsv, opsv)
        valid_stat.append(valid_loss, valid_acc, valid_f1_score, valid_prec, valid_rec, valid_roc_auc, duration)

        if valid_acc > best_acc:
            best_acc = valid_acc
            util.save_checkpoint(bert, True, best_model_file)
            tfpr, ttpr, _ = roc_curve(targs, ops)
            train_stat.update_best(ttpr, tfpr, train_acc, i)
            vfpr, vtpr, _ = roc_curve(targsv, opsv)
            valid_stat.update_best(vtpr, vfpr, best_acc, i)

        if args.track:
            log_wandb(wandb, targs, ops, targsv, opsv, train_stat, valid_stat, i)

        print("\n{}\t{:06.8f}\t{:06.8f}\t{:06.8f}\t{:06.8f}\t{:06.8f}\t{:06.8f}".format(i+1, train_acc*100, train_loss, 
                                                    train_f1_score, valid_acc*100, 
                                                    valid_loss, valid_f1_score))

    print("Summary of best run::")
    print("Best Accuracy: {}".format(valid_stat.best_acc))
    print("Roc Auc score: {}".format(valid_stat.roc_aucs[valid_stat.best_epoch]))
    print("Loss: {}".format(valid_stat.losses[valid_stat.best_epoch]))
    print("Area Under Curve: {}".format(auc(valid_stat.fprs, valid_stat.tprs)))

    if args.save_plot:
        util.plot(train_stat, valid_stat)
    
    # Now Load best model and get predictions
    test_dataset = dataset.BertDataset(
        text=test_df.text.values,
        tokenizer= tokenizer,
        max_len= max_len
    )

    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        1,
        shuffle=False,
        num_workers=1
    )

    util.load_checkpoint(bert, best_model_file)
    _, opst, _ = train.test(test_dl, bert, criterion, device)

    sub_csv = pd.DataFrame(columns=['id', 'target'])
    sub_csv['id'] = test_df['id']
    sub_csv['target'] = opst

    csv_name =  "./" + str(model_name) + "_sub.csv"
    sub_csv.to_csv(csv_name, index=False)


if __name__ == "__main__":
    run()