import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import auc
import torch

class AvgStats(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.losses =[]
        self.accs =[]
        self.f1 = []
        self.precs = []
        self.recs = []
        self.roc_aucs = []
        self.tprs = []
        self.fprs = []
        self.its = []
        self.best_acc = 0.0
        self.best_epoch = -1
        
    def append(self, loss, acc, f1, prec, rec, roc_auc, it):
        self.losses.append(loss)
        self.accs.append(acc)
        self.f1.append(f1)
        self.precs.append(prec)
        self.recs.append(rec)
        self.roc_aucs.append(roc_auc)
        self.its.append(it)
    
    def update_best(self, tpr, fpr, acc, epoch):
        self.best_acc = acc
        self.tprs = tpr
        self.fprs = fpr
        self.best_epoch = epoch


def save_checkpoint(model, is_best, filename='./model/checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        torch.save(model.state_dict(), filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")

def load_checkpoint(model, filename = './model/checkpoint.pth.tar'):
    sd = torch.load(filename, map_location=lambda storage, loc: storage)
    names = set(model.state_dict().keys())
    for n in list(sd.keys()): 
        if n not in names and n+'_raw' in names:
            if n+'_raw' not in sd: sd[n+'_raw'] = sd[n]
            del sd[n]
    model.load_state_dict(sd)

def plot(train_stats, valid_stats):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,15))
    axs[0,0].set_title("Loss Train vs Valid")
    axs[0,0].plot(train_stats.losses, 'r', label='Train')
    axs[0,0].plot(valid_stats.losses, 'g', label='Valid')
    axs[0,0].set_ylabel("Loss")
    axs[0,0].set_xlabel("Epochs")
    axs[0,0].set_xticks([i for i in range(0, len(train_stats.losses))]) 
    axs[0,0].legend()

    axs[0,1].set_title("Accuracy Train vs Valid")
    axs[0,1].plot(train_stats.accs, 'r', label='Train')
    axs[0,1].plot(valid_stats.accs, 'g', label='Valid')
    axs[0,1].set_ylabel("Accuracy")
    axs[0,1].set_xlabel("Epochs")
    axs[0,1].set_xticks([i for i in range(0, len(train_stats.accs))])
    axs[0,1].legend()

    axs[1,0].set_title("Precision vs Recall")
    axs[1,0].plot(train_stats.recs, train_stats.precs, 'r', label='Train')
    axs[1,0].plot(valid_stats.recs, valid_stats.precs, 'g', label='Valid')
    axs[1,0].set_ylabel("Precision")
    axs[1,0].set_xlabel("Recall")
    axs[1,0].legend()

    axs[1,1].set_title("Roc/Auc curve")
    tauc = auc(train_stats.fprs, train_stats.tprs)
    vauc = auc(valid_stats.fprs, valid_stats.tprs)
    axs[1,1].plot(train_stats.fprs, train_stats.tprs, 'r', label='Train AUC: {}'.format(tauc))
    axs[1,1].plot(valid_stats.fprs, valid_stats.tprs, 'g', label='Valid AUC: {}'.format(vauc))
    axs[1,1].set_ylabel("TPR")
    axs[1,1].set_xlabel("FPR")
    axs[1,1].legend()

    plt.savefig('./plot.jpg')
    plt.show()



