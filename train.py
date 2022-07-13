import argparse
import yaml
import pandas as pd 
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from munch import Munch
from data.prepare_data import pre_process, load_dataset
from models.graph_residual_nw import MTL2
from train_func import *
from utils import *


#===============================================================
def train(args):
    if args.pre_process_data:
        print("pre processing data")
        all_smiles, list_labels = pre_process(args.path_raw, args.tasks)
        trainset, valset, testset = load_dataset(path=args.path_processed, all_smiles=all_smiles, list_labels=list_labels, seed=args.seed)
    else:
        print("Loading dataset")
        trainset, valset, testset = load_dataset(path=args.path_processed, seed=args.seed)
    train_loader  = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    val_loader    = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    test_loader   = DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    #---------------------------------------
    # Parameter
    n_epcoh = args.epochs
    learning_rate = args.learning_rate
    layers = args.layers
    num_classes_tasks = len(args.tasks)
    filt_sizes = args.filt_sizes
    # Model
    model = MTL2(layers, num_classes_tasks, filt_sizes)
    model = model.to(args.device)
    criterion = nn.BCELoss()
    # Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #---------------------------------------
    print('size of training set: ', len(trainset))
    print('size of validation set: ', len(valset))
    print('size of test set: ', len(testset))
    #---------------------------------------
    print("Training Model")
    training_loss_list   = []
    validation_loss_list = []
    val_loss_check = 10
    for epoch in range(n_epcoh):
        train_results = train_funct(epoch, model, optimizer, criterion, args.tasks, train_loader)
        validation_results = validate(epoch, model, criterion, args.tasks, val_loader)
        training_loss_list.append(train_results)
        validation_loss_list.append(validation_results[0])
        #---------------------------------------
        if validation_results[0] < val_loss_check:
            val_loss_check = validation_results[0]
            torch.save(model.state_dict(), args.save_ckpt)
            test_performance = test(epoch, model, args.tasks, test_loader)
            val_performance  = validation_results[1]
        print('#---------------------------------------------------------')
    #---------------------------------------    
    print("Save result")  
    df = pd.DataFrame({'Task'     : args.tasks,
                'test_AUC'        : test_performance[0],
                'test_PR_AUC'     : test_performance[1],
                'test_ACC'        : test_performance[2],
                'test_BA'         : test_performance[3],
                'test_MCC'        : test_performance[4],
                'test_CK'         : test_performance[5],
                'test_sensitivity': test_performance[6],
                'test_specificity': test_performance[7],
                'test_precision'  : test_performance[8],
                'test_F1'         : test_performance[9]})
    df.to_csv(args.path_result + 'result.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', default='./config/config.yaml', help='path to yaml config file', type=argparse.FileType('r'))
    parser.add_argument('--no_cuda', action='store_true', help='Use CPU')

    parsed_args = parser.parse_args()
    with parsed_args.config as f:
        params = yaml.load(f)
    args = parse_args(Munch(params), **vars(parsed_args))
    train(args)
    