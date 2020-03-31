import torch
import pickle
import os

class AvgStats(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.losses =[]
        self.precs =[]
        self.f1 = []
        self.its = []
        
    def append(self, loss, prec, f1, it):
        self.losses.append(loss)
        self.precs.append(prec)
        self.f1.append(f1)
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

def save_stats(stats, filepath):
    with open(filepath, 'w') as fp:
        pickle.dump(stats, filepath)

def load_stats(filepath):
    assert(os.path.exists(filepath))
    with open(filepath, 'r') as fp:
        stats = pickle.loads(fp.read())
    return stats