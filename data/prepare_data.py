import os
import torch
import pandas as pd
from data.utils import get_encode, MultiDataset
from sklearn.model_selection import train_test_split

#===============================================================
def pre_process(path, tasks):
    '''
    Data are read from a csv file corresponding to each task to obtain SMILES and labels (of compound sets) of each task.
    Then, for a single task, labels for SMILES(s) not belong to the task are added. If a compound not belong to any task,
    its label will be assigned "6".
    Input
        path: folder containing data files of each task 
        tasks: list of tasks 
    Output
        all_smiles: list of SMILES(s) 
        list_labels: list of labels of SMILES(s) corresponding to each task
    '''
    # Get smiles and labels for each task 
    task_smiles = []
    task_labels = []
    all_smiles  = []
    for task in tasks:
        path_task = path + "/refined_merged_{}.csv".format(task)
        data      = pd.read_csv(path_task)
        smiles    = data['SMILES'].tolist()
        label     = data['Label'].tolist()
        task_smiles.append(smiles)
        task_labels.append(label)
        all_smiles.extend(smiles)
    #---------------------------------------
    # labeling for all smiles 
    all_smiles  = list(set(all_smiles))
    list_labels = [[] for i in range(len(tasks))]
    for i in range(len(tasks)):
        for smiles in all_smiles:
            if smiles in task_smiles[i]:
                idx = task_smiles[i].index(smiles)
                list_labels[i].append(task_labels[i][idx])
            else:
                # smiles in not labeled in this task
                list_labels[i].append(6)
    return all_smiles, list_labels


#===============================================================
def load_dataset(data_path, all_smiles=None, list_labels=None, seed=2):
    # Load processed dataset 
    train_path = data_path + '/train_seed_{}.ckpt'.format(seed)
    val_path   = data_path + '/val_seed_{}.ckpt'.format(seed)
    test_path  = data_path + '/test_seed_{}.ckpt'.format(seed)
    #---------------------------------------
    if os.path.isfile(train_path) and os.path.isfile(val_path) and os.path.isfile(test_path):
        train = torch.load(train_path)
        val   = torch.load(val_path)
        test  = torch.load(test_path)
        return train, val, test
    #---------------------------------------
    if all_smiles==None or list_labels==None:
        print("You need to run pre process data: change the value of 'pre_process_data' in the config.yaml True !")
        return
    data_list = get_encode(all_smiles, list_labels)
    X_, X_test = train_test_split(data_list, test_size=0.1, random_state=2)
    X_train, X_val = train_test_split(X_, test_size=0.15, random_state=seed)
    #---------------------------------------
    train = MultiDataset(root=data_path, data_list=X_train, dataset="train")
    val   = MultiDataset(root=data_path, data_list=X_val, dataset="val")
    test  = MultiDataset(root=data_path, data_list=X_test, dataset="test")
    print("Num_of_sample train_set={}, val_set={}, test_set={}".format(len(train), len(val), len(test)))
    #---------------------------------------
    # Save dataset 
    torch.save(train, train_path)
    torch.save(val, val_path)
    torch.save(test, test_path)
    #---------------------------------------
    return load_dataset(data_path, all_smiles, list_labels, seed)

