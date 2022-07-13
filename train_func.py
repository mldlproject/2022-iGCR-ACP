import numpy as np
import time
import torch
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, cohen_kappa_score, balanced_accuracy_score


#
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

 
def get_loss_task(criterion, num_classes_tasks, probs, labels, device):
    total_loss = 0
    for t_id in range(num_classes_tasks):
        y_pred = probs[t_id] # output of each task
        y_label = labels[:, t_id:t_id+1].squeeze() # label of task
        validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0] # index tại những dữ liệu tồn tại label
        if len(validId) == 0:
            continue
        if y_label.dim() == 0:
            y_label = y_label.unsqueeze(0)

        y_pred = y_pred[torch.tensor(validId).to(device)]
        y_label = y_label[torch.tensor(validId).to(device)]

        loss = criterion(y_pred.view(-1).squeeze(), y_label.squeeze())
        total_loss += loss
    total_loss = total_loss/num_classes_tasks
    return total_loss

def get_prob_task(num_classes_tasks, probs, labels, device):
    prob_list = []
    label_list = []
    for t_id in range(num_classes_tasks):
        y_pred = probs[t_id] # output of each task
        y_label = labels[:, t_id:t_id+1].squeeze() # label of task
        validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0] # index tại những dữ liệu tồn tại label
        if len(validId) == 0:
            continue
        if y_label.dim() == 0:
            y_label = y_label.unsqueeze(0)

        y_pred = y_pred[torch.tensor(validId).to(device)]
        y_label = y_label[torch.tensor(validId).to(device)]
        
        prob_list.append(y_pred.detach().cpu().view_as(y_label).numpy()) # lưu giá trị dự đoán 
        label_list.append(y_label.detach().cpu().numpy()) # lưu label dữ liệu
     
    return prob_list, label_list

def get_performace(y_label_list, y_pred_list, tasks):
    trn_roc =  np.array([roc_auc_score(y_label_list[i], y_pred_list[i]) for i in range(len(tasks))])
    trn_prc =  np.array([metrics.auc(precision_recall_curve(y_label_list[i], y_pred_list[i])[1],
                        precision_recall_curve(y_label_list[i], y_pred_list[i])[0]) for i in range(len(tasks))])
    #---------------------------------------------
    #--------------------------------------------
    predicted_labels = [[] for i in range(len(tasks))]
    for i in range(len(tasks)):
        for prob in y_pred_list[i]: 
            predicted_labels[i].append(np.round(prob))

    # acc = accuracy_score(labels, predicted_labels)
    trn_acc = np.array([accuracy_score(y_label_list[i], predicted_labels[i]) for i in range(len(tasks))])
    trn_ba  = np.array([balanced_accuracy_score(y_label_list[i], predicted_labels[i]) for i in range(len(tasks))])
    trn_mcc = np.array([matthews_corrcoef(y_label_list[i], predicted_labels[i]) for i in range(len(tasks))])
    trn_ck  =  np.array([cohen_kappa_score(y_label_list[i], predicted_labels[i]) for i in range(len(tasks))])
    trn_sensitivity, trn_specificity, trn_precision, trn_f1 = [], [], [], []
    for i in range(len(tasks)):
        tn, fp, fn, tp = confusion_matrix(y_label_list[i], predicted_labels[i]).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision   = tp / (tp + fp)
        f1 = 2*precision*sensitivity / (precision + sensitivity)
        trn_sensitivity.append(sensitivity)
        trn_specificity.append(specificity)
        trn_precision.append(precision)
        trn_f1.append(f1)

    trn_sensitivity, trn_specificity, trn_precision, trn_f1 = np.array(trn_sensitivity), np.array(trn_specificity), np.array(trn_precision), np.array(trn_f1)

    perform = [trn_roc, trn_prc, trn_acc, trn_ba, trn_mcc, trn_ck, trn_sensitivity, trn_specificity, trn_precision, trn_f1]
    return perform

##########################################################################################                 
###                  DEFINE TRAINING, VALIDATION, AND TEST FUNCTION                    ###           
##########################################################################################
# Training Function
def train_funct(epoch, model, optimizer, criterion, tasks, train_loader):
    model.train()
    train_loss = 0
    start_time = time.time()
    for batch_idx, batch in enumerate(train_loader):
        data       = batch.to(device)
        labels     = batch.y.to(device).float() # Chứa label của 12 task
        
        outputs = model(data)
        #------------------- 
        optimizer.zero_grad()
        #------------------- 
        avg_loss = get_loss_task(criterion, len(tasks), outputs, labels, device)  

        avg_loss.backward()
        train_loss += avg_loss.item()*len(data) #(loss.item is the average loss of training batch)
        optimizer.step() 

    #------------------- 
    print('====> Epoch: {}, training time {},  Average Train Loss: {:.4f}'.format(epoch, time.time() - start_time, train_loss / len(train_loader)))
    train_loss = (train_loss / len(train_loader.dataset) )

    return train_loss

##########################################################################################
# Validation Function
def validate(epoch, model, criterion, tasks, val_loader):
    model.eval()
    validation_loss = 0
    pred_prob = []
    y_pred_list = {}
    y_label_list = {}
    for i in range(len(tasks)):
        y_label_list[i] = []
        y_pred_list[i] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            data       = batch.to(device)
            labels     = batch.y.to(device).float() # Chứa label của 12 task
            
            outputs = model(data)

            avg_loss = get_loss_task(criterion, len(tasks), outputs, labels, device)
            validation_loss += avg_loss.item()*len(data) #(loss.item is the average loss of training batch)

            pred_list, label_list = get_prob_task(len(tasks), outputs, labels, device) # Chưa predict của 5 task

            # print(len(label_list))
            for i in range(len(tasks)):
                if label_list[i] != 'None':
                    y_label_list[i].extend(label_list[i])
                    y_pred_list[i].extend(pred_list[i])
        
    print('====> Epoch: {} Average Validation Loss: {:.4f}'.format(epoch, validation_loss / len(val_loader)))
    validation_loss = (validation_loss / len(val_loader.dataset) )
    perform = get_performace(y_label_list, y_pred_list, tasks)

    return validation_loss, perform
    
# ##########################################################################################
# Test Function
def test(current_iter, model, tasks, test_loader):
    model.eval()
    y_pred_list = {}
    y_label_list = {}
    for i in range(len(tasks)):
        y_label_list[i] = []
        y_pred_list[i] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data       = batch.to(device)
            labels     = batch.y.to(device).float() # Chứa label của 12 task
            outputs = model(data)
            pred_list, label_list = get_prob_task(len(tasks), outputs, labels, device) # Chưa predict của 5 task

            # print(len(label_list))
            for i in range(len(tasks)):
                if label_list[i] != 'None':

                    y_label_list[i].extend(label_list[i])
                    y_pred_list[i].extend(pred_list[i])
    perform = get_performace(y_label_list, y_pred_list, tasks)
    print("Performance of model at epoch {} on test dataset".format(current_iter))
    print("AUC of {} task: {}".format(len(tasks),perform[0])) 
    print("PR_AUC of {} task: {}".format(len(tasks),perform[1]))
    print("######################################################")
    return perform

