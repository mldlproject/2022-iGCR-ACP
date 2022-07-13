import os
import numpy as np
import torch
from munch import Munch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score


#===============================================================
def parse_args(args, **kwargs):
    args = Munch({"epoch": 0}, **args)
    kwargs = Munch({"no_cuda": False, "debug": False}, **kwargs)
    args.device = "cuda" if torch.cuda.is_available() and not kwargs.no_cuda else "cpu"
    
    if "decoder_args" not in args or args.decoder_args is None:
        args.decoder_args = {}
    if "model_path" in args:
        args.out_path = os.path.join(args.model_path, args.name)
        os.makedirs(args.out_path, exist_ok=True)
    return args


#===============================================================
def printPerformance(labels, probs, threshold:float=None, decimal:int=None, printout=False):
    if threshold != None:
        assert threshold < 0 and threshold > 1, "threshold must be in the range [0 to 1]"
        predicted_labels = np.array([1 if prob >= threshold else 0 for prob in probs])
    else:
        predicted_labels = np.round(probs)
    #------------------------------------------------    
    tn, fp, fn, tp = confusion_matrix(labels, predicted_labels).ravel()
    if decimal != None:
        assert decimal <= 8, "decimal must be int and in the range [0 to 8]"
        d = decimal
    else:
        d = 8
    aucroc = round(roc_auc_score(labels, probs), d)
    aucpr  = round(average_precision_score(labels, probs), d)
    acc    = round(accuracy_score(labels, predicted_labels), d) 
    ba     = round(balanced_accuracy_score(labels, predicted_labels), d)
    mcc    = round(matthews_corrcoef(labels, predicted_labels), d)
    sen    = round(tp / (tp + fn), d)
    spe    = round(tn / (tn + fp), d)
    pre    = round(tp / (tp + fp), d)
    f1     = round(2*pre*sen / (pre + sen), d)
    ck     = round(cohen_kappa_score(labels, predicted_labels), d)
    #------------------------------------------------
    if printout:
        print('AUCROC: {}'.format(aucroc))
        print('AUCPR: {}'.format(aucpr))
        print('ACC: {}'.format(acc))
        print('BA: {}'.format(ba))
        print('SN/RE : {}'.format(sen))
        print('SP: {}'.format(spe))
        print('PR: {}'.format(pre))
        print('MCC: {}'.format(mcc))
        print('F1: {}'.format(f1))
        print('CK: {}'.format(ck))
    #------------------------------------------------
    return aucroc, aucpr, acc, ba, sen, spe, pre, mcc, f1, ck