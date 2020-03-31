import matplotlib.pyplot as plt
import os
import pickle
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
        self.its = []
        
    def append(self, loss, acc, f1, prec, rec, roc_auc, it):
        self.losses.append(loss)
        self.accs.append(acc)
        self.f1.append(f1)
        self.precs.append(prec)
        self.recs.append(rec)
        self.roc_aucs.append(roc_auc)
        self.its.append(it)


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
    fig= plt.figure()
    fig, axs = plt.subplots(3, 2)
    axs[0,0].set_title("Loss Train vs Valid")
    axs[0,0].plot(train_stats.losses, 'r', label='Train')
    axs[0,0].plot(valid_stats.losses, 'g', label='Valid')
    axs[0,0].set_ylabel("Loss")
    axs[0,0].legend()
    
    fig, axs = plt.subplots(3, 2)
    axs[0,1].set_title("Accuracy Train vs Valid")
    axs[0,1].plot(train_stats.accs, 'r', label='Train')
    axs[0,1].plot(valid_stats.accs, 'g', label='Valid')
    axs[0,1].set_ylabel("Accuracy")
    axs[0,1].legend()

    fig, axs = plt.subplots(3, 2)
    axs[1,0].set_title("Precision Train vs Valid")
    axs[1,0].plot(train_stats.precs, 'r', label='Train')
    axs[1,0].plot(valid_stats.precs, 'g', label='Valid')
    axs[1,0].set_ylabel("Precision")
    axs[1,0].legend()

    fig, axs = plt.subplots(3, 2)
    axs[1,1].set_title("Recall Train vs Valid")
    axs[1,1].plot(train_stats.recs, 'r', label='Train')
    axs[1,1].plot(valid_stats.recs, 'g', label='Valid')
    axs[1,1].set_ylabel("Recall")
    axs[1,1].legend()

    fig, axs = plt.subplots(3, 2)
    axs[2,0].set_title("Roc/Auc score Train vs Valid")
    axs[2,0].plot(train_stats.roc_aucs, 'r', label='Train')
    axs[2,0].plot(valid_stats.roc_aucs, 'g', label='Valid')
    axs[2,0].set_ylabel("ROC AUC")
    axs[2,0].legend()

    for ax in axs.flat:
        ax.set(xlabel='Epochs')
        ax.label_outer()

    fig.savefig('./plot.jpg')

    plt.show()



